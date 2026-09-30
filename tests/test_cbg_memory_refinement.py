"""Exact execution optimizations: bounded workspace and refinement equivalence."""
import unittest
import gc
import weakref
from dataclasses import replace
from collections import OrderedDict
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from src.cbg.dual import (
    _RefinementState, _refine_center, _refine_center_with_state,
    _relative_residual,
)
from src.cbg.functionals import make_benign_functional, make_safety_functional
from src.cbg.geometry import MetricProduct, functional_jvp, metric_product_with_jvp
from src.cbg.spectral_grad import (
    compute_spectral_center, direct_spectral_gradient, rescale_owned_spectral_center_,
    streamed_loss_gradient,
    compact_spectral_center, centered_spectral_gradient,
)
from src.cbg.tree_ops import (
    accumulate_detached_gradient, tree_add, tree_dot, tree_dot_fp64_detached,
    tree_gram, tree_linear_combination, tree_norm, tree_scale, tree_sub,
)
from src.cbg.validate import fixture


def literal_old_refinement(fh, fb, p, center, mu):
    """Pre-optimization implementation, used only on a tiny model."""
    with torch.no_grad():
        residual, _, _ = _relative_residual(center)
        for _ in range(2):
            residual = tree_add(residual, center.direction,
                                alpha=-tree_dot(center.b_direction, residual) / center.b)
        correction = compute_spectral_center(fh, fb, p, tree_scale(residual, 1 / tree_norm(residual)), mu=mu)
        vectors = (center.direction, correction.direction)
        double = lambda t: OrderedDict((n, v.double()) for n, v in t.items())
        vv = tuple(double(v) for v in vectors)
        h = tree_gram(vv, tuple(double(c.gh_direction) for c in (center, correction)))
        b = tree_gram(vv, tuple(double(c.b_direction) for c in (center, correction)))
        h, b = (h+h.T)/2, (b+b.T)/2
        chol = torch.linalg.cholesky(b)
        left = torch.linalg.solve_triangular(chol, h, upper=False)
        reduced = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        _, ev = torch.linalg.eigh((reduced+reduced.T)/2)
        coefficients = torch.linalg.solve_triangular(chol.T, ev[:, -1:], upper=True).flatten()
        direction = tree_linear_combination(vectors, coefficients)
        raw = compute_spectral_center(fh, fb, p, direction, mu=mu)
        scale = raw.b.rsqrt()
        if tree_dot(direction, center.direction) < 0:
            scale = -scale
        return compute_spectral_center(fh, fb, p, tree_scale(direction, scale), mu=mu,
            safety_product=MetricProduct(tree_scale(raw.gh_direction, scale), raw.zh*scale),
            benign_product=MetricProduct(tree_scale(raw.gb_direction, scale), raw.zb*scale))


class MemoryRefinementTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_refinement_equivalence_gradient_and_reduced_operator_work(self):
        for dtype in (torch.float64, torch.float32):
            model, p, harmful, safe, benign = fixture(torch.device("cpu"))
            model.to(dtype)
            fh = make_safety_functional(model, harmful, safe, chunk_size=2)
            fb = make_benign_functional(model, benign, chunk_size=2, token_blocks=2)
            gen = torch.Generator().manual_seed(91)
            u = OrderedDict((n, torch.randn(v.shape, dtype=dtype, generator=gen)) for n, v in p.items())
            before = {n: (v.detach().clone(), v.data_ptr()) for n, v in p.items()}
            center = compute_spectral_center(fh, fb, p, u, mu=.1)
            expected = literal_old_refinement(fh, fb, p, center, .1)
            with patch("src.cbg.spectral_grad.metric_product_with_jvp", wraps=metric_product_with_jvp) as products:
                with patch("src.cbg.dual.functional_jvp", wraps=functional_jvp) as responses:
                    actual = _refine_center(fh, fb, p, center, .1)
            self.assertEqual(products.call_count, 2)  # Previously four metric products.
            self.assertEqual(responses.call_count, 2)  # Correction JVPs, no correction VJPs.
            tol = 2e-5 if dtype == torch.float32 else 1e-10
            torch.testing.assert_close(actual.rho, expected.rho, rtol=tol, atol=tol)
            torch.testing.assert_close(actual.b, torch.ones_like(actual.b), rtol=tol, atol=tol)
            for n in u:
                torch.testing.assert_close(actual.direction[n], expected.direction[n], rtol=tol, atol=tol)
            g = direct_spectral_gradient(fh, fb, p, actual.direction, mu=.1, center=actual).gradient
            ref = direct_spectral_gradient(fh, fb, p, expected.direction, mu=.1, center=expected).gradient
            self.assertLess(float(tree_norm(tree_sub(g, ref))/tree_norm(ref)), tol)
            for n, v in p.items():
                self.assertTrue(torch.equal(v, before[n][0]))
                self.assertEqual(v.data_ptr(), before[n][1])
            self.assertTrue(all(v.grad_fn is None for v in g.values()))

    def test_refinement_retains_accepted_residual_subspace(self):
        """Multiple corrections must enlarge one Ritz space, not restart it."""

        model, p, harmful, safe, benign = fixture(torch.device("cpu"))
        model.float()
        fh = make_safety_functional(model, harmful, safe, chunk_size=2)
        fb = make_benign_functional(model, benign, chunk_size=2, token_blocks=2)
        generator = torch.Generator().manual_seed(0)
        direction = OrderedDict(
            (name, torch.randn(value.shape, dtype=value.dtype, generator=generator))
            for name, value in p.items()
        )
        center = compute_spectral_center(fh, fb, p, direction, mu=.1)
        center = rescale_owned_spectral_center_(center, center.b.rsqrt())
        state = _RefinementState(
            vectors=(center.direction,),
            safety_responses=(center.zh,),
            benign_responses=(center.zb,),
        )
        residuals = [float(_relative_residual(center)[2])]
        for expected_size in (2, 3, 4):
            candidate, expanded = _refine_center_with_state(
                fh, fb, p, center, .1, state
            )
            self.assertIsNotNone(candidate)
            candidate_residual = float(_relative_residual(candidate)[2])
            self.assertLess(candidate_residual, residuals[-1])
            self.assertEqual(len(expanded.vectors), expected_size)
            self.assertTrue(all(
                value.grad_fn is None
                for vector in expanded.vectors
                for value in vector.values()
            ))
            center, state = candidate, expanded
            residuals.append(candidate_residual)
        self.assertLess(residuals[-1], residuals[0] / 100)

    def test_fp64_dot_bounded_conversion_and_noncontiguous_leaves(self):
        class AllocationShapes(TorchDispatchMode):
            def __init__(self):
                self.max_double = 0
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                for t in tree_flatten(result)[0]:
                    if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
                        self.max_double = max(self.max_double, t.numel())
                return result
        a = OrderedDict(w=torch.arange(1105, dtype=torch.float32).reshape(65,17).T.requires_grad_())
        b = OrderedDict(w=torch.cos(a["w"]).detach())
        with AllocationShapes() as shapes:
            actual = tree_dot_fp64_detached(a, b, chunk_elements=64)
        expected = (a["w"].double()*b["w"].double()).sum()
        torch.testing.assert_close(actual, expected)
        self.assertLessEqual(shapes.max_double, 64)
        self.assertIsNone(actual.grad_fn)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                tree_dot_fp64_detached(a, b, chunk_elements=invalid)

    def test_accumulator_owns_storage_even_when_gradient_aliases_parameter(self):
        p = OrderedDict(w=torch.arange(5., requires_grad=True))
        before = p["w"].detach().clone()
        acc = accumulate_detached_gradient(None, p)
        ptr = acc["w"].data_ptr()
        self.assertNotEqual(ptr, p["w"].data_ptr())
        acc = accumulate_detached_gradient(acc, p)
        self.assertEqual(ptr, acc["w"].data_ptr())
        torch.testing.assert_close(acc["w"], 2*before)
        self.assertTrue(torch.equal(before, p["w"]))
        scaled = accumulate_detached_gradient(None, p, alpha=.25)
        torch.testing.assert_close(scaled["w"], .25*before)
        scaled = accumulate_detached_gradient(scaled, p, alpha=-.5)
        torch.testing.assert_close(scaled["w"], -.25*before)
        with self.assertRaises(ValueError):
            accumulate_detached_gradient(p, p)

    def test_owned_center_normalization_reuses_storage(self):
        model, p, harmful, safe, benign = fixture(torch.device("cpu"))
        fh = make_safety_functional(model, harmful, safe, chunk_size=2)
        fb = make_benign_functional(model, benign, chunk_size=2, token_blocks=2)
        gen = torch.Generator().manual_seed(17)
        u = OrderedDict((n, torch.randn(v.shape, dtype=v.dtype, generator=gen)) for n, v in p.items())
        raw = compute_spectral_center(fh, fb, p, u, mu=.1)
        scale = raw.b.rsqrt()
        expected = compute_spectral_center(
            fh, fb, p, tree_scale(raw.direction, scale), mu=.1,
            safety_product=MetricProduct(tree_scale(raw.gh_direction, scale), raw.zh*scale),
            benign_product=MetricProduct(tree_scale(raw.gb_direction, scale), raw.zb*scale),
        )
        pointers = tuple(v.data_ptr() for tree in (
            raw.direction, raw.gh_direction, raw.gb_direction, raw.b_direction,
        ) for v in tree.values())
        actual = rescale_owned_spectral_center_(raw, scale)
        self.assertEqual(pointers, tuple(v.data_ptr() for tree in (
            actual.direction, actual.gh_direction, actual.gb_direction, actual.b_direction,
        ) for v in tree.values()))
        torch.testing.assert_close(actual.b, expected.b)
        torch.testing.assert_close(actual.rho, expected.rho)
        for left, right in zip(actual.direction.values(), expected.direction.values()):
            torch.testing.assert_close(left, right)
        self.assertTrue(all(v.grad_fn is None for tree in (
            actual.direction, actual.gh_direction, actual.gb_direction, actual.b_direction,
        ) for v in tree.values()))

    def test_streamed_preservation_matches_joint_gradient(self):
        p = OrderedDict(w=torch.tensor([.2, -.7, 1.1], dtype=torch.float64, requires_grad=True))
        def stream():
            for fraction, power in ((.2, 2), (.3, 3), (.5, 4)):
                yield fraction, p["w"].pow(power).sum()
        value, gradient = streamed_loss_gradient(stream(), p)
        joint = sum(f*loss for f, loss in stream())
        expected, = torch.autograd.grad(joint, (p["w"],))
        torch.testing.assert_close(value, joint.detach())
        torch.testing.assert_close(gradient["w"], expected)
        self.assertIsNone(gradient["w"].grad_fn)

    def test_external_cotangent_and_shared_gradient_storage_remain_unchanged(self):
        p = OrderedDict(w=torch.tensor([.2, -.7, 1.1], dtype=torch.float64, requires_grad=True))
        coefficient = torch.tensor([2., 3., -1.], dtype=torch.float64)
        before = coefficient.clone()
        def losses():
            yield 1., (p["w"] * coefficient).sum()
            yield 1., p["w"].square().sum()
        _, gradient = streamed_loss_gradient(losses(), p)
        torch.testing.assert_close(coefficient, before, rtol=0, atol=0)
        torch.testing.assert_close(gradient["w"], before + 2*p["w"].detach())
        p["v"] = p["w"].detach().clone().requires_grad_()
        def shared_losses():
            yield 1., (p["w"] + p["v"]).sum()
            yield 1., p["w"].square().sum() + 3*p["v"].square().sum()
        _, gradient = streamed_loss_gradient(shared_losses(), p)
        torch.testing.assert_close(gradient["w"], 1+2*p["w"].detach())
        torch.testing.assert_close(gradient["v"], 1+6*p["v"].detach())
        self.assertNotEqual(gradient["w"].data_ptr(), gradient["v"].data_ptr())

    def test_tree_combination_preserves_first_and_second_derivatives(self):
        x = torch.tensor([.3, -.8], dtype=torch.float64, requires_grad=True)
        coefficients = torch.tensor([.4, .7], dtype=torch.float64, requires_grad=True)
        vectors = (OrderedDict(w=x.square()), OrderedDict(w=x.pow(3)))
        result = tree_linear_combination(vectors, coefficients)["w"].sum()
        gx, gc = torch.autograd.grad(result, (x, coefficients), create_graph=True)
        torch.testing.assert_close(gx, 2*coefficients[0]*x + 3*coefficients[1]*x.square())
        torch.testing.assert_close(gc, torch.stack((x.square().sum(), x.pow(3).sum())))
        h, = torch.autograd.grad(gx.sum(), (x,))
        torch.testing.assert_close(h, 2*coefficients[0] + 6*coefficients[1]*x)

    def test_compact_center_releases_products_preserves_both_derivatives_and_cache(self):
        model, p, harmful, safe, benign = fixture(torch.device("cpu"))
        fh = make_safety_functional(model, harmful, safe, chunk_size=1)
        fb = make_benign_functional(model, benign, chunk_size=1, token_blocks=2)
        u = OrderedDict((n, torch.ones_like(v)) for n, v in p.items())
        center = compute_spectral_center(fh, fb, p, u, mu=.1)
        direct = direct_spectral_gradient(fh, fb, p, u, mu=.1, center=center).gradient
        fd = centered_spectral_gradient(fh, fb, p, u, mu=.1, center=center).gradient
        refs = [weakref.ref(v) for tree in (center.gh_direction, center.gb_direction, center.b_direction)
                for v in tree.values()]
        compact = compact_spectral_center(center)
        self.assertIsNotNone(center.gh_direction)
        del center
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        for actual, expected in (
            (direct_spectral_gradient(fh, fb, p, u, mu=.1, center=compact).gradient, direct),
            (centered_spectral_gradient(fh, fb, p, u, mu=.1, center=compact).gradient, fd),
        ):
            for n in p:
                torch.testing.assert_close(actual[n], expected[n], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "compact"):
            rescale_owned_spectral_center_(compact, 1.)
        with torch.no_grad():
            next(iter(p.values())).add_(.01)
        with self.assertRaisesRegex(ValueError, "stale"):
            direct_spectral_gradient(fh, fb, p, u, mu=.1, center=compact)

    def test_normalization_rejects_offset_parameter_alias_and_bad_scale_before_mutation(self):
        model, p, harmful, safe, benign = fixture(torch.device("cpu"))
        fh = make_safety_functional(model, harmful, safe)
        fb = make_benign_functional(model, benign)
        u = OrderedDict((n, torch.ones_like(v)) for n, v in p.items())
        center = compute_spectral_center(fh, fb, p, u, mu=.1)
        before = {n: v.clone() for n, v in p.items()}
        for scale in (0., float("nan"), float("inf"), torch.ones(2)):
            with self.assertRaisesRegex(ValueError, "finite nonzero scalar"):
                rescale_owned_spectral_center_(center, scale)
        alias = OrderedDict(center.direction)
        name = next(iter(p))
        alias[name] = p[name].detach().reshape(-1)[1:]
        with self.assertRaisesRegex(ValueError, "aliases"):
            rescale_owned_spectral_center_(replace(center, direction=alias), 2.)
        for n in p:
            torch.testing.assert_close(p[n], before[n], rtol=0, atol=0)
            torch.testing.assert_close(center.direction[n], torch.ones_like(p[n]), rtol=0, atol=0)

    def test_trainer_releases_solver_products_before_outer_ad(self):
        from test_cbg_optimized import OptimizedCbgTest
        fixture_test = OptimizedCbgTest()
        fixture_test.setUp()
        trainer, _ = fixture_test.trainer_fixture(cbg_version="v5", geometry_solver="dual",
            spectral_derivative_backend="direct_rof", benign_token_blocks=2)
        original_search = trainer._search_geometry
        refs = []
        def search(*args, **kwargs):
            result = original_search(*args, **kwargs)
            refs.extend(weakref.ref(v) for tree in (result.center.gh_direction,
                result.center.gb_direction, result.center.b_direction) for v in tree.values())
            return result
        def derivative(*args, **kwargs):
            gc.collect()
            self.assertTrue(refs and all(ref() is None for ref in refs))
            self.assertIsNone(kwargs["center"].gh_direction)
            return direct_spectral_gradient(*args, **kwargs)
        with patch.object(trainer, "_search_geometry", side_effect=search), patch(
            "src.core.cbg_trainer.direct_spectral_gradient", side_effect=derivative,
        ):
            trainer.training_step(fixture_test.model, dict(harmful=fixture_test.harmful,
                safe=fixture_test.safe, benign=fixture_test.benign))

    def test_bounded_fallback_releases_failed_search_products(self):
        from src.cbg.bounded import bounded_rayleigh_solve
        from src.cbg.dual import dual_rayleigh_solve
        model, p, harmful, safe, benign = fixture(torch.device("cpu"))
        fh = make_safety_functional(model, harmful, safe)
        fb = make_benign_functional(model, benign, token_blocks=2)
        refs = []
        def center(*args, **kwargs):
            result = compute_spectral_center(*args, **kwargs)
            refs.extend(weakref.ref(v) for tree in (result.gh_direction,
                result.gb_direction, result.b_direction) for v in tree.values())
            return result
        def fallback(*args, **kwargs):
            gc.collect()
            self.assertTrue(refs and all(ref() is None for ref in refs))
            return dual_rayleigh_solve(*args, **kwargs)
        with patch("src.cbg.bounded.compute_spectral_center", side_effect=center), patch(
            "src.cbg.bounded.dual_rayleigh_solve", side_effect=fallback,
        ):
            result = bounded_rayleigh_solve(fh, fb, p, mu=.1, max_enrichments=0,
                residual_tolerance=1e-12, generator=torch.Generator().manual_seed(13))
        self.assertTrue(result.diagnostics.dual_fallback)
        self.assertGreater(result.diagnostics.basis_size, 0)


if __name__ == "__main__":
    unittest.main()
