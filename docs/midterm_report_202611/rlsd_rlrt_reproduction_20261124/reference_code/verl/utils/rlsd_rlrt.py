"""Detached token credit weights for answer-conditioned RLSD/RLRT experiments."""

import math
from typing import Optional

import torch


GRPO_OPSD_MODES = frozenset({"rlsd", "rlrt", "hybrid", "grpo"})
_MAX_LOG_WEIGHT = 60.0


def teacher_is_active(mode: str, lam: float) -> bool:
    """Use only rank-independent configuration to decide whether to score teacher."""
    if mode not in GRPO_OPSD_MODES:
        raise ValueError(f"Unknown grpo_opsd_mode={mode!r}; expected {sorted(GRPO_OPSD_MODES)}")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"RLSD/RLRT lambda must be finite and in [0, 1], got {lam}")
    return mode != "grpo" and lam > 0.0


def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    # Float64 accumulation keeps diagnostic means finite for extreme finite inputs.
    selected = torch.where(mask, values.detach(), 0).double()
    return (selected.sum() / mask.sum().clamp_min(1)).item()


def _negative_response_rows(advantages: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    negative = valid & (advantages < 0)
    rows = negative.any(dim=-1, keepdim=True)
    if (rows & (valid & (advantages >= 0)).any(dim=-1, keepdim=True)).any():
        raise ValueError("Negative EAPO requires uniformly negative advantages on valid response tokens")
    return rows


@torch.no_grad()
def build_negative_eapo_weights(
    *,
    old_token_entropies: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kappa: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Redistribute only negative sequence credit using frozen old-policy entropy.

    Call once on the complete rollout batch, before splitting it for updates.
    Entropy must be full-vocabulary Shannon entropy, not sampled surprisal.
    Valid negative-response weights have mean one; other responses retain one.
    Masked positions are zero, including unknown groups supplied with zero mask.
    """
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"Negative EAPO kappa must be finite and nonnegative, got {kappa}")
    if old_token_entropies.ndim != 2 or not old_token_entropies.is_floating_point():
        raise ValueError("old_token_entropies must be a floating [B, T] tensor")
    if response_mask.shape != old_token_entropies.shape or response_mask.device != old_token_entropies.device:
        raise ValueError("response_mask must match old_token_entropies shape and device")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only zero or one")
    if advantages.device != old_token_entropies.device or not advantages.is_floating_point():
        raise ValueError("advantages must be floating point on the old_token_entropies device")
    if advantages.ndim == 1 and advantages.shape[0] == old_token_entropies.shape[0]:
        advantages = advantages.unsqueeze(-1)
    try:
        advantages = torch.broadcast_to(advantages, old_token_entropies.shape)
    except RuntimeError as exc:
        raise ValueError("advantages must broadcast to old_token_entropies shape") from exc
    valid = response_mask.bool()
    if not torch.isfinite(advantages[valid]).all():
        raise ValueError("Nonfinite advantages on valid response tokens")
    dtype = torch.float64 if old_token_entropies.dtype == torch.float64 else torch.float32
    weights = valid.to(dtype)
    negative_rows = (valid & (advantages < 0)).any(dim=-1, keepdim=True)
    q_low = q_high = 0.0
    if kappa > 0.0:
        negative_rows = _negative_response_rows(advantages, valid)
        entropies = old_token_entropies[valid].double()
        if not torch.isfinite(entropies).all() or (entropies < 0).any():
            raise ValueError("Valid old_token_entropies must be finite and nonnegative")
        if entropies.numel():
            quantiles = torch.quantile(entropies, torch.tensor([0.1, 0.9], device=entropies.device, dtype=torch.float64))
            q_low, q_high = quantiles[0].item(), quantiles[1].item()
            safe_entropies = torch.where(valid, old_token_entropies, 0).double()
            normalized = ((safe_entropies - quantiles[0]) / (quantiles[1] - quantiles[0] + 1e-8)).clamp(0, 1)
            log_raw = (-kappa * normalized).masked_fill(~valid, -torch.inf)
            row_max = log_raw.max(dim=-1, keepdim=True).values
            row_max = torch.where(valid.any(dim=-1, keepdim=True), row_max, 0)
            raw = (log_raw - row_max).exp()
            row_mean = raw.sum(dim=-1, keepdim=True) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
            allocated = raw / torch.where(row_mean > 0, row_mean, 1)
            weights = torch.where(negative_rows, allocated.to(dtype), weights)
    negative_tokens = valid & negative_rows
    metrics = {
        "grpo_opsd/eapo_negative_kappa": float(kappa),
        "grpo_opsd/eapo_entropy_q10": q_low,
        "grpo_opsd/eapo_entropy_q90": q_high,
        "grpo_opsd/eapo_negative_weight_mean": _mean(weights, negative_tokens),
        "grpo_opsd/eapo_negative_weight_min": weights[negative_tokens].min().item() if negative_tokens.any() else 0.0,
        "grpo_opsd/eapo_negative_weight_max": weights[negative_tokens].max().item() if negative_tokens.any() else 0.0,
        "grpo_opsd/eapo_negative_response_fraction": _mean(negative_rows.float(), valid.any(dim=-1, keepdim=True)),
    }
    return weights.detach(), metrics


def build_rlsd_rlrt_advantages(
    *,
    teacher_log_probs: Optional[torch.Tensor],
    student_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    lam: float,
    clip_range: float,
    positive_clip_range: Optional[float] = None,
    negative_clip_range: Optional[float] = None,
    mode: str = "rlsd",
    negative_only: bool = False,
    thought_mask: Optional[torch.Tensor] = None,
    negative_eapo_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reweight advantages while leaving the PPO current/old policy ratio intact.

    RLSD uses exp(sign(A) * delta). RLRT uses exp(-delta) on positive A
    and one on negative A. Hybrid uses RLRT on positive A and RLSD on
    negative A. Correctness corresponds to sign(A) only for binary-reward
    GRPO groups containing both outcomes; homogeneous groups have A == 0.
    Branch clip ranges fall back to clip_range when not explicitly configured.

    Teacher work is disabled only by mode/lambda, never by local advantages.
    Masked padding is ignored, and all valid inputs must be finite.
    """
    active = teacher_is_active(mode, lam)
    positive_clip_range = clip_range if positive_clip_range is None else positive_clip_range
    negative_clip_range = clip_range if negative_clip_range is None else negative_clip_range
    for name, value in (("clip_range", clip_range),
                        ("positive_clip_range", positive_clip_range),
                        ("negative_clip_range", negative_clip_range)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
    if response_mask.shape != student_log_probs.shape or response_mask.device != student_log_probs.device:
        raise ValueError("response_mask must match student_log_probs shape and device")
    if advantages.device != student_log_probs.device:
        raise ValueError("advantages must match student_log_probs device")
    try:
        # The original helper also accepts one sequence advantage per row [B, 1].
        advantages = torch.broadcast_to(advantages, student_log_probs.shape)
    except RuntimeError as exc:
        raise ValueError("advantages must broadcast to student_log_probs shape") from exc
    if not advantages.is_floating_point() or not student_log_probs.is_floating_point():
        raise ValueError("advantages and student_log_probs must be floating point")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only zero or one")
    valid = response_mask.bool()
    for name, value in (("advantages", advantages), ("student_log_probs", student_log_probs)):
        if not torch.isfinite(value[valid]).all():
            raise ValueError(f"Nonfinite {name} on valid response tokens")
    allowed = valid.clone()
    if thought_mask is not None:
        if thought_mask.shape != advantages.shape or thought_mask.device != advantages.device:
            raise ValueError("thought_mask must match advantages shape and device")
        if not torch.all((thought_mask == 0) | (thought_mask == 1)):
            raise ValueError("thought_mask must contain only zero or one")
        allowed &= thought_mask.bool()
    safe_advantages = torch.where(valid, advantages, 0)
    if negative_only:
        allowed &= safe_advantages.detach().double().sum(dim=-1, keepdim=True) < 0
    positive = valid & (safe_advantages > 0)
    negative = valid & (safe_advantages < 0)
    selected = allowed & (positive | negative)
    if mode == "rlrt":
        selected &= positive
    work_dtype = torch.float64 if student_log_probs.dtype == torch.float64 else torch.float32
    delta = torch.zeros_like(student_log_probs, dtype=work_dtype)
    log_weights = torch.zeros_like(delta)
    if active:
        if teacher_log_probs is None:
            raise ValueError("Active RLSD/RLRT requires teacher_log_probs")
        if teacher_log_probs.shape != advantages.shape or teacher_log_probs.device != advantages.device:
            raise ValueError("teacher_log_probs must match advantages shape and device")
        if not teacher_log_probs.is_floating_point() or not torch.isfinite(teacher_log_probs[valid]).all():
            raise ValueError("Nonfinite or non-floating teacher_log_probs on valid response tokens")
        student = torch.where(valid, student_log_probs.detach(), 0).to(work_dtype)
        teacher = torch.where(valid, teacher_log_probs.detach(), 0).to(work_dtype)
        # A finite-input subtraction can still overflow. Saturate before exp;
        # saturation preserves every result after clipping to [1-eps, 1+eps].
        delta = (teacher - student).clamp(-_MAX_LOG_WEIGHT, _MAX_LOG_WEIGHT)
        if mode == "rlsd":
            proposed = safe_advantages.detach().sign() * delta
        else:
            proposed = -delta
        log_weights = torch.where(selected, proposed, 0)
    else:
        selected = torch.zeros_like(valid)
    clip_ranges = torch.full_like(log_weights, negative_clip_range).masked_fill(positive, positive_clip_range)
    low_log, high_log = torch.log1p(-clip_ranges), torch.log1p(clip_ranges)
    low_clipped = selected & (log_weights < low_log)
    high_clipped = selected & (log_weights > high_log)
    raw_weights = log_weights.clamp(-_MAX_LOG_WEIGHT, _MAX_LOG_WEIGHT).exp()
    # Clip in log space before exponentiation; no overflowing intermediate exp.
    clipped_weights = log_weights.clamp(low_log, high_log).exp()
    effective_lambda = lam if active else 0.0
    factors = (1.0 - effective_lambda) + effective_lambda * clipped_weights
    if negative_eapo_weights is not None:
        if student_log_probs.ndim != 2:
            raise ValueError("Negative EAPO requires [B, T] student_log_probs")
        if negative_eapo_weights.shape != advantages.shape or negative_eapo_weights.device != advantages.device:
            raise ValueError("negative_eapo_weights must match advantages shape and device")
        if not negative_eapo_weights.is_floating_point():
            raise ValueError("negative_eapo_weights must be floating point")
        eapo_weights = torch.where(valid, negative_eapo_weights.detach(), 0).to(work_dtype)
        if not torch.isfinite(eapo_weights).all() or (eapo_weights < 0).any():
            raise ValueError("negative_eapo_weights must be finite and nonnegative on valid response tokens")
        # Unit weights bypass arithmetic so disabled EAPO exactly reproduces
        # the original positive/negative values and their gradient multipliers.
        changed_rows = (valid & (eapo_weights != 1)).any(dim=-1, keepdim=True)
        if changed_rows.any():
            negative_rows = _negative_response_rows(safe_advantages, valid)
            fused_rows = changed_rows & negative_rows
            counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            baseline_mean = torch.where(valid, factors, 0).sum(dim=-1, keepdim=True) / counts
            product = factors * eapo_weights
            product_mean = product.sum(dim=-1, keepdim=True) / counts
            if (fused_rows & (product_mean <= 0)).any():
                raise ValueError("negative_eapo_weights must have a positive mean on each negative response")
            fused = baseline_mean * product / torch.where(product_mean > 0, product_mean, 1)
            factors = torch.where(fused_rows & valid, fused, factors)
    weighted_advantages = safe_advantages * factors.detach()
    if not torch.isfinite(weighted_advantages).all():
        raise ValueError("Nonfinite weighted advantages")
    metrics = {
        "grpo_opsd/stgca_delta_mean": _mean(delta, valid),
        "grpo_opsd/stgca_weight_mean": _mean(raw_weights, valid),
        "grpo_opsd/stgca_clipped_weight_mean": _mean(clipped_weights, valid),
        "grpo_opsd/stgca_clip_low_ratio": _mean(low_clipped.float(), valid),
        "grpo_opsd/stgca_clip_high_ratio": _mean(high_clipped.float(), valid),
        "grpo_opsd/stgca_adv_mean": _mean(weighted_advantages, valid),
        "grpo_opsd/stgca_adv_abs_mean": _mean(weighted_advantages.abs(), valid),
        "grpo_opsd/stgca_effective_lambda": effective_lambda,
        "grpo_opsd/teacher_forward_skipped": float(not active),
        "grpo_opsd/zero_advantage_fraction": _mean((safe_advantages == 0).float(), valid),
        "grpo_opsd/positive_reweight_clip_range": float(positive_clip_range),
        "grpo_opsd/negative_reweight_clip_range": float(negative_clip_range),
    }
    for branch, branch_mask in (("positive", positive), ("negative", negative)):
        metrics[f"grpo_opsd/{branch}_weight_mean"] = _mean(factors, branch_mask)
        metrics[f"grpo_opsd/{branch}_clipped_fraction"] = _mean(
            (low_clipped | high_clipped).float(), branch_mask
        )
        metrics[f"grpo_opsd/{branch}_token_count"] = float(branch_mask.sum().item())
    if negative_only:
        metrics["grpo_opsd/stgca_negative_only"] = 1.0
        metrics["grpo_opsd/stgca_neg_seq_ratio"] = (negative.any(dim=-1).float().mean().item())
        metrics["grpo_opsd/stgca_neg_delta_mean"] = _mean(delta, negative)
    if thought_mask is not None:
        metrics["grpo_opsd/stgca_thought_only"] = 1.0
        metrics["grpo_opsd/stgca_thought_token_ratio"] = _mean(thought_mask.float(), valid)
    return weighted_advantages, metrics
