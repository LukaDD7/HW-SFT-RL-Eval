"""CPU math tests plus execution of the actual actor control flow with fake workers."""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rlsd_rlrt", ROOT / "verl/utils/rlsd_rlrt.py")
weights_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights_module)
build = weights_module.build_rlsd_rlrt_advantages


class WeightTests(unittest.TestCase):
    def calculate(self, mode="rlsd", delta=None, advantages=None, **kwargs):
        if delta is None:
            delta = torch.tensor([[math.log(2.0), -math.log(2.0)], [math.log(2.0), -math.log(2.0)]])
        if advantages is None:
            advantages = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
        values = dict(
            teacher_log_probs=delta - 5.0,
            student_log_probs=torch.full_like(delta, -5.0),
            advantages=advantages,
            response_mask=torch.ones_like(delta),
            lam=0.5,
            clip_range=0.2,
            mode=mode,
        )
        values.update(kwargs)
        return build(**values)

    def test_hand_calculated_modes(self):
        expected = {
            "rlsd": [[2.2, 1.8], [-1.8, -2.2]],
            "rlrt": [[1.8, 2.2], [-2.0, -2.0]],
            "hybrid": [[1.8, 2.2], [-1.8, -2.2]],
            "grpo": [[2.0, 2.0], [-2.0, -2.0]],
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                result, metrics = self.calculate(mode)
                torch.testing.assert_close(result, torch.tensor(values))
                self.assertEqual(metrics["grpo_opsd/teacher_forward_skipped"], float(mode == "grpo"))

    def test_unclipped_ratio_and_interpolation(self):
        delta = torch.tensor([[math.log(1.1), math.log(0.9)]], dtype=torch.float64)
        adv = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        result, _ = self.calculate(delta=delta, advantages=adv, lam=0.25)
        torch.testing.assert_close(result, torch.tensor([[1.025, -(0.75 + 0.25 / 0.9)]], dtype=torch.float64))

    def test_positive_clip_strength_does_not_change_negative_branch(self):
        result, metrics = self.calculate("hybrid", positive_clip_range=0.5, negative_clip_range=0.2)
        torch.testing.assert_close(result, torch.tensor([[1.5, 2.5], [-1.8, -2.2]]))
        self.assertEqual(metrics["grpo_opsd/positive_reweight_clip_range"], 0.5)
        self.assertEqual(metrics["grpo_opsd/negative_reweight_clip_range"], 0.2)
        result, _ = self.calculate("rlrt", positive_clip_range=0.5, negative_clip_range=0.2)
        torch.testing.assert_close(result, torch.tensor([[1.5, 2.5], [-2.0, -2.0]]))

    def test_independent_clips_preserve_masks_and_zero_advantages(self):
        result, _ = self.calculate(
            "hybrid", positive_clip_range=0.5, negative_clip_range=0.2,
            thought_mask=torch.tensor([[1, 0], [1, 0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[1.5, 2.0], [-1.8, -2.0]]))
        result, _ = self.calculate("hybrid", positive_clip_range=0.5, negative_only=True)
        torch.testing.assert_close(result, torch.tensor([[2.0, 2.0], [-1.8, -2.2]]))
        for lam in (0.0, 0.5):
            result, _ = self.calculate(
                "hybrid", positive_clip_range=0.5, advantages=torch.zeros(2, 2), lam=lam,
                teacher_log_probs=None if lam == 0 else torch.full((2, 2), -5.0),
            )
            self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_equal_branch_overrides_reproduce_legacy_result(self):
        for mode in ("rlsd", "rlrt", "hybrid", "grpo"):
            baseline, _ = self.calculate(mode)
            override, _ = self.calculate(mode, positive_clip_range=0.2, negative_clip_range=0.2)
            torch.testing.assert_close(override, baseline, rtol=0, atol=0)

    def test_legacy_rlsd_matches_finite_upstream_formula(self):
        generator = torch.Generator().manual_seed(7)
        delta = torch.randn((5, 17), generator=generator)
        adv = torch.randn((5, 17), generator=generator)
        expected = adv * (0.5 + 0.5 * torch.exp(adv.sign() * delta).clamp(0.8, 1.2))
        actual, _ = self.calculate(delta=delta, advantages=adv)
        torch.testing.assert_close(actual, expected)

    def test_zero_lambda_needs_no_teacher(self):
        for mode in ("rlsd", "rlrt", "hybrid", "grpo"):
            actual, metrics = self.calculate(mode, lam=0.0, teacher_log_probs=None)
            torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0], [-2.0, -2.0]]))
            self.assertEqual(metrics["grpo_opsd/teacher_forward_skipped"], 1.0)

    def test_grpo_needs_no_teacher_even_with_positive_lambda(self):
        actual, metrics = self.calculate("grpo", teacher_log_probs=None)
        self.assertEqual(metrics["grpo_opsd/stgca_effective_lambda"], 0.0)
        torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0], [-2.0, -2.0]]))

    def test_zero_advantages_stay_exactly_zero(self):
        for mode in ("rlsd", "rlrt", "hybrid", "grpo"):
            actual, metrics = self.calculate(mode, advantages=torch.zeros(2, 2))
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
            self.assertEqual(metrics["grpo_opsd/zero_advantage_fraction"], 1.0)

    def test_sequence_advantages_broadcast_like_upstream(self):
        result, _ = self.calculate(advantages=torch.tensor([[2.0], [-2.0]]))
        torch.testing.assert_close(result, torch.tensor([[2.2, 1.8], [-1.8, -2.2]]))

    def test_extreme_finite_logs_never_overflow(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            limit = torch.finfo(dtype).max
            teacher = torch.tensor([[limit, -limit], [-limit, limit]], dtype=dtype)
            student = -teacher
            for mode in ("rlsd", "rlrt", "hybrid"):
                actual, metrics = self.calculate(
                    mode, delta=torch.zeros(2, 2, dtype=dtype),
                    teacher_log_probs=teacher, student_log_probs=student,
                )
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_masked_nonfinite_padding_is_ignored(self):
        actual, metrics = self.calculate(
            "hybrid", advantages=torch.tensor([[2.0, float("nan")], [-2.0, float("inf")]]),
            student_log_probs=torch.tensor([[-5.0, float("nan")], [-5.0, float("inf")]]),
            teacher_log_probs=torch.tensor([[-4.0, float("inf")], [-4.0, float("nan")]]),
            response_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )
        torch.testing.assert_close(actual, torch.tensor([[1.8, 0.0], [-1.8, 0.0]]))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_empty_mask_metrics_are_finite(self):
        actual, metrics = self.calculate(response_mask=torch.zeros(2, 2))
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_weights_detached_from_student_and_teacher(self):
        teacher = torch.full((2, 2), -4.95, requires_grad=True)
        student = torch.full((2, 2), -5.0, requires_grad=True)
        adv = torch.tensor([[2.0, 2.0], [-2.0, -2.0]], requires_grad=True)
        actual, _ = self.calculate("hybrid", teacher_log_probs=teacher, student_log_probs=student, advantages=adv)
        actual.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(student.grad)
        torch.testing.assert_close(adv.grad, torch.full_like(adv, 0.5 + 0.5 * math.exp(-0.05)))

    def test_legacy_negative_only_and_thought_mask(self):
        actual, _ = self.calculate(negative_only=True, thought_mask=torch.tensor([[1, 1], [1, 0]]))
        torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0], [-1.8, -2.0]]))

    def test_branch_metrics_reflect_applied_weights(self):
        _, metrics = self.calculate("rlrt")
        self.assertAlmostEqual(metrics["grpo_opsd/negative_weight_mean"], 1.0)
        self.assertEqual(metrics["grpo_opsd/negative_clipped_fraction"], 0.0)
        self.assertEqual(metrics["grpo_opsd/positive_clipped_fraction"], 1.0)

    def test_validation(self):
        for overrides in (
            {"lam": float("nan")}, {"lam": 1.1}, {"lam": -0.1},
            {"clip_range": float("inf")}, {"clip_range": 1.0}, {"clip_range": -0.1},
            {"positive_clip_range": float("nan")}, {"positive_clip_range": 1.0},
            {"negative_clip_range": float("inf")}, {"negative_clip_range": -0.1},
            {"mode": "unknown"}, {"teacher_log_probs": None},
            {"advantages": torch.full((2, 2), float("nan"))},
            {"teacher_log_probs": torch.full((2, 2), float("inf"))},
            {"student_log_probs": torch.full((2, 2), float("nan"))},
            {"response_mask": torch.full((2, 2), 0.5)},
            {"thought_mask": torch.full((2, 2), float("nan"))},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.calculate(**overrides)


def _actor_method():
    """Load the real method without importing Ray, FlashAttention, or FSDP."""
    path = ROOT / "verl/workers/actor/dp_opsd_actor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelOPSDActor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_policy_grpo_opsd")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)

    def append(metrics, values):
        for key, value in values.items():
            metrics[key].append(value)

    namespace = dict(
        torch=torch, np=np, Literal=Literal, defaultdict=defaultdict, Any=object, DataProto=object,
        dcsd_progress=lambda *args, **kwargs: None,
        teacher_is_active=weights_module.teacher_is_active, append_to_dict=append,
        dist=SimpleNamespace(all_reduce=lambda *a, **k: None, ReduceOp=SimpleNamespace(SUM=None)),
        prepare_dynamic_batch=lambda batch, **kwargs: ([batch], None),
    )
    # Exercise upstream PPO and masked means, including empty local masks.
    functional_tree = ast.parse((ROOT / "verl/utils/torch_functional.py").read_text(encoding="utf-8"))
    masked_mean = next(n for n in functional_tree.body if isinstance(n, ast.FunctionDef) and n.name == "masked_mean")
    exec(compile(ast.Module(body=[masked_mean], type_ignores=[]), "torch_functional.py", "exec"), namespace)
    namespace["VF"] = SimpleNamespace(masked_mean=namespace["masked_mean"])
    core_tree = ast.parse((ROOT / "verl/trainer/core_algos.py").read_text(encoding="utf-8"))
    loss_functions = [n for n in core_tree.body if isinstance(n, ast.FunctionDef) and n.name in {"average_loss", "compute_policy_loss"}]
    exec(compile(ast.Module(body=loss_functions, type_ignores=[]), "core_algos.py", "exec"), namespace)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


class _Batch:
    def __init__(self, mode, lam, active, zero=False):
        self.meta_info = {"temperature": 1.0, "opsd_config": {
            "grpo_opsd_mode": mode, "stgca_lambda": lam, "global_step": 10,
            "stgca_teacher_sync_interval": 10,
        }}
        self.batch = {name: torch.ones(1, 2) for name in (
            "input_ids", "attention_mask", "position_ids", "responses", "response_mask", "old_log_probs"
        )}
        self.batch["advantages"] = torch.zeros(1, 2) if zero else torch.ones(1, 2)
        self.batch["old_log_probs"].fill_(-5.0)
        if active:
            for suffix in ("input_ids", "attention_mask", "position_ids"):
                self.batch[f"grpo_opsd_teacher_{suffix}"] = torch.ones(1, 2)
        self.non_tensor_batch = {"multi_modal_inputs": None}

    def select(self, keys, non_tensor_keys):
        for key in keys:
            if key not in self.batch:
                raise AssertionError(f"Requested absent tensor {key}")
        return self

    def split(self, size):
        return [self]


class ActorControlTests(unittest.TestCase):
    def make_actor(self, dynamic=False):
        actor = SimpleNamespace(
            config=SimpleNamespace(
                clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_dual=3.0,
                global_batch_size_per_device=1, micro_batch_size_per_device_for_update=1,
                ppo_epochs=1, dynamic_batching=dynamic, max_token_len_per_gpu=None,
                use_kl_loss=False, loss_type="default", loss_avg_mode="token",
            ),
            actor_module=torch.nn.Module(), rank=1, world_size=1,
            teacher_calls=0, sync_calls=0,
        )
        actor._get_teacher_module = lambda: torch.nn.Module()
        actor.student_lp = torch.full((1, 2), -5.0, requires_grad=True)
        actor._forward_micro_batch = lambda *a, **k: actor.student_lp

        def teacher(*args, **kwargs):
            actor.teacher_calls += 1
            return torch.full((1, 2), -4.95)

        def sync():
            actor.sync_calls += 1

        actor._forward_micro_batch_with_module = teacher
        actor._sync_teacher_from_actor = sync
        actor._build_stgca_advantages = build
        actor._optimizer_step = lambda: torch.tensor(0.0)
        return actor

    def test_inactive_paths_omit_teacher_keys_forward_and_sync(self):
        method = _actor_method()
        for mode, lam in (("rlsd", 0.0), ("rlrt", 0.0), ("hybrid", 0.0), ("grpo", 0.5)):
            for dynamic in (False, True):
                actor = self.make_actor(dynamic)
                metrics = method(actor, _Batch(mode, lam, active=False))
                self.assertEqual(actor.teacher_calls, 0)
                self.assertEqual(actor.sync_calls, 0)
                self.assertEqual(metrics["grpo_opsd/teacher_forward_skipped"], [1.0])

    def test_local_zero_advantages_do_not_skip_fsdp_teacher(self):
        actor = self.make_actor()
        _actor_method()(actor, _Batch("hybrid", 0.5, active=True, zero=True))
        self.assertEqual(actor.teacher_calls, 1)
        self.assertEqual(actor.sync_calls, 1)

    def test_loss_mask_excludes_invalid_tokens_from_denominator(self):
        actor = self.make_actor()
        batch = _Batch("grpo", 0.5, active=False)
        batch.batch["rlsd_loss_mask"] = torch.tensor([[1.0, 0.0]])
        _actor_method()(actor, batch)
        torch.testing.assert_close(actor.student_lp.grad, torch.tensor([[-1.0, 0.0]]))

    def test_actor_applies_positive_clip_override_to_ppo_gradient(self):
        actor = self.make_actor()
        actor._forward_micro_batch_with_module = lambda *a, **k: torch.tensor(
            [[-5.0 + math.log(2), -5.0 - math.log(2)]]
        )
        batch = _Batch("hybrid", 0.5, active=True)
        batch.meta_info["opsd_config"].update(
            rlsd_positive_reweight_clip_range=0.5, rlsd_negative_reweight_clip_range=0.2
        )
        metrics = _actor_method()(actor, batch)
        torch.testing.assert_close(actor.student_lp.grad, torch.tensor([[-0.375, -0.625]]))
        self.assertEqual(metrics["grpo_opsd/positive_reweight_clip_range"], [0.5])

    def test_empty_local_loss_mask_keeps_forward_and_backward_finite(self):
        actor = self.make_actor()
        batch = _Batch("hybrid", 0.5, active=True)
        batch.batch["rlsd_loss_mask"] = torch.zeros(1, 2)
        metrics = _actor_method()(actor, batch)
        self.assertEqual(actor.teacher_calls, 1)
        self.assertTrue(torch.equal(actor.student_lp.grad, torch.zeros(1, 2)))
        self.assertTrue(all(math.isfinite(v) for values in metrics.values() for v in values))

    def test_reactivation_refreshes_teacher_before_reuse(self):
        actor = self.make_actor()
        method = _actor_method()
        method(actor, _Batch("hybrid", 0.0, active=False))
        metrics = method(actor, _Batch("hybrid", 0.5, active=True))
        self.assertEqual(actor.sync_calls, 2)  # Reactivation refresh, then periodic step-10 sync.
        self.assertEqual(metrics["grpo_opsd/teacher_reactivated_sync"], [1.0])


class ConfigTests(unittest.TestCase):
    def config_class(self):
        path = ROOT / "verl/trainer/opsd_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OPSDConfig")
        namespace = dict(
            dataclass=dataclass, field=field, Optional=Optional, math=math,
            get_abs_path=lambda path, **kwargs: path,
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["OPSDConfig"]

    def test_default_and_alias_preserve_original_rlsd(self):
        config = self.config_class()(stgca_lambda=0.25)
        config.post_init()
        self.assertEqual(config.grpo_opsd_mode, "rlsd")
        self.assertFalse(config.binary_reward_groups)
        self.assertEqual(config.rlsd_lambda, 0.25)

    def test_modes_and_binary_group_switch(self):
        for mode in ("rlsd", "rlrt", "hybrid", "grpo"):
            config = self.config_class()(grpo_opsd_mode=mode, binary_reward_groups=True)
            config.post_init()
            self.assertEqual(config.grpo_opsd_mode, mode)

    def test_failure_only_eapo_requires_binary_simple_grpo(self):
        config = self.config_class()(
            eapo_negative_kappa=math.log(4), enable_grpo_opsd_simple=True, binary_reward_groups=True,
        )
        config.post_init()
        self.assertEqual(config.eapo_negative_kappa, math.log(4))
        for kwargs in (
            {"eapo_negative_kappa": math.log(4)},
            {"eapo_negative_kappa": math.log(4), "enable_grpo_opsd_simple": True},
            {"eapo_negative_kappa": -1}, {"eapo_negative_kappa": float("nan")},
            {"eapo_negative_kappa": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config_class()(**kwargs).post_init()

    def test_independent_clip_config_survives_alias_resolution(self):
        config = self.config_class()(
            stgca_reweight_clip_range=0.2, rlsd_positive_reweight_clip_range=0.5,
            rlsd_negative_reweight_clip_range=0.2,
        )
        config.post_init()
        self.assertEqual(config.rlsd_reweight_clip_range, 0.2)
        self.assertEqual(config.rlsd_positive_reweight_clip_range, 0.5)
        self.assertEqual(config.rlsd_negative_reweight_clip_range, 0.2)

    def test_config_rejects_invalid_mode_and_nonfinite_weights(self):
        for kwargs in (
            {"grpo_opsd_mode": "invalid"}, {"rlsd_lambda": float("nan")},
            {"rlsd_reweight_clip_range": float("nan")}, {"rlsd_reweight_clip_range": 1.0},
            {"rlsd_positive_reweight_clip_range": float("nan")},
            {"rlsd_positive_reweight_clip_range": 1.0},
            {"rlsd_negative_reweight_clip_range": float("inf")},
            {"rlsd_negative_reweight_clip_range": -0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config_class()(**kwargs).post_init()


if __name__ == "__main__":
    unittest.main()
