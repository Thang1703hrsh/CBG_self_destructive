from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from src.cbg.tree_ops import (
    assert_same_structure,
    restrict_tree,
    tree_add,
    tree_clone,
    tree_detach,
    tree_dot,
    tree_norm,
    tree_scale,
    tree_sub,
    tree_zeros_like,
)


class TreeOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = OrderedDict(
            first=torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True),
            second=torch.tensor([[-1.0, 3.0]], dtype=torch.float64, requires_grad=True),
        )
        self.b = OrderedDict(
            first=torch.tensor([4.0, -2.0], dtype=torch.float64),
            second=torch.tensor([[5.0, 1.0]], dtype=torch.float64),
        )

    def test_dot_norm_scale_add_and_subtract(self) -> None:
        expected_dot = sum((self.a[name] * self.b[name]).sum() for name in self.a)
        torch.testing.assert_close(tree_dot(self.a, self.b), expected_dot)
        expected_norm = torch.sqrt(sum(value.square().sum() for value in self.a.values()))
        torch.testing.assert_close(tree_norm(self.a), expected_norm)

        scaled = tree_scale(self.a, 2.5)
        added = tree_add(self.a, self.b, alpha=0.25)
        subtracted = tree_sub(self.a, self.b)
        for name in self.a:
            torch.testing.assert_close(scaled[name], 2.5 * self.a[name])
            torch.testing.assert_close(added[name], self.a[name] + 0.25 * self.b[name])
            torch.testing.assert_close(subtracted[name], self.a[name] - self.b[name])

    def test_clone_detach_and_zeros_like_preserve_order_and_shapes(self) -> None:
        cloned = tree_clone(self.a)
        detached = tree_detach(self.a)
        zeros = tree_zeros_like(self.a)
        self.assertEqual(tuple(cloned), tuple(self.a))
        self.assertEqual(tuple(detached), tuple(self.a))
        self.assertEqual(tuple(zeros), tuple(self.a))
        for name in self.a:
            self.assertNotEqual(cloned[name].data_ptr(), self.a[name].data_ptr())
            self.assertFalse(detached[name].requires_grad)
            torch.testing.assert_close(zeros[name], torch.zeros_like(self.a[name]))

    def test_restrict_tree_uses_explicit_requested_order(self) -> None:
        restricted = restrict_tree(self.a, ("second", "first"))
        self.assertEqual(tuple(restricted), ("second", "first"))
        self.assertIs(restricted["second"], self.a["second"])

    def test_structure_mismatch_fails_loudly(self) -> None:
        wrong_order = OrderedDict(reversed(tuple(self.b.items())))
        with self.assertRaises(ValueError):
            assert_same_structure(self.a, wrong_order)
        with self.assertRaises(KeyError):
            restrict_tree(self.a, ("missing",))


if __name__ == "__main__":
    unittest.main()
