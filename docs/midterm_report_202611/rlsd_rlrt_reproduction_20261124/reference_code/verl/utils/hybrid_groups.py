"""Question-level binary reward diagnostics, independent of rollout row ordering."""
from collections import defaultdict

import torch


def binary_group_mask(uids, correctness, judgeable, expected_group_size):
    if len(uids) != len(correctness) or len(uids) != len(judgeable):
        raise ValueError("Reward diagnostics must have one entry per rollout row")
    groups = defaultdict(list)
    for row, uid in enumerate(uids):
        groups[str(uid)].append(row)
    mask = torch.ones(len(uids), dtype=torch.float32)
    counts = {key: 0 for key in ("mixed", "all_correct", "all_wrong", "unjudgeable")}
    for indices in groups.values():
        if len(indices) != expected_group_size:
            raise ValueError(f"Incomplete rollout group: {len(indices)} != {expected_group_size}")
        valid = all(float(judgeable[i]) == 1.0 for i in indices)
        if not valid:
            counts["unjudgeable"] += 1
            mask[indices] = 0
            continue
        values = [float(correctness[i]) for i in indices]
        if any(x not in (0.0, 1.0) for x in values):
            raise ValueError("Hybrid experiments require binary correctness rewards")
        total = sum(values)
        counts["all_wrong" if total == 0 else "all_correct" if total == len(indices) else "mixed"] += 1
    n = len(groups)
    metrics = {f"groups/{key}_count": value for key, value in counts.items()}
    metrics.update({f"groups/{key}_fraction": value / max(n, 1) for key, value in counts.items()})
    metrics["groups/total_count"] = n
    metrics["groups/effective_question_count"] = counts["mixed"]
    metrics["groups/effective_trajectory_count"] = counts["mixed"] * expected_group_size
    return mask, metrics


def effective_lambda(step, target, warmup, decay, mode):
    if mode == "grpo":
        return 0.0
    if warmup > 0 and step < warmup:
        return target * step / warmup
    if decay > 0:
        return target * max(1.0 - (step - warmup) / decay, 0.0)
    return target
