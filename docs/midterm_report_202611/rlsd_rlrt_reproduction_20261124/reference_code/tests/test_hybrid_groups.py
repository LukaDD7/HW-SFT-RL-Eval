import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("hybrid_groups", Path(__file__).parents[1] / "verl/utils/hybrid_groups.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class GroupTests(unittest.TestCase):
    def test_reordered_groups_and_unknown(self):
        mask, metrics = mod.binary_group_mask(["a", "b", "a", "b", "c", "c"], [1, 0, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1], 2)
        self.assertEqual(mask.tolist(), [1, 0, 1, 0, 1, 1])
        self.assertEqual(metrics["groups/mixed_count"], 1)
        self.assertEqual(metrics["groups/unjudgeable_count"], 1)
        self.assertEqual(metrics["groups/all_correct_count"], 1)
        self.assertEqual(metrics["groups/effective_trajectory_count"], 2)

    def test_nonbinary_and_partial_fail(self):
        with self.assertRaises(ValueError): mod.binary_group_mask(["x", "x"], [.9, 0], [1, 1], 2)
        with self.assertRaises(ValueError): mod.binary_group_mask(["x"], [1], [1], 2)

    def test_schedule(self):
        self.assertEqual(mod.effective_lambda(25, .5, 0, 50, "hybrid"), .25)
        self.assertEqual(mod.effective_lambda(50, .5, 0, 50, "hybrid"), 0)
        self.assertEqual(mod.effective_lambda(100, .5, 0, 0, "hybrid"), .5)
        self.assertEqual(mod.effective_lambda(1, .5, 0, 0, "grpo"), 0)

    def test_extended_teacher_schedule_boundaries(self):
        for step, expected in ((0, .5), (1, .495), (50, .25), (99, .005), (100, 0), (150, 0)):
            with self.subTest(step=step):
                self.assertAlmostEqual(mod.effective_lambda(step, .5, 0, 100, "hybrid"), expected)


if __name__ == "__main__": unittest.main()
