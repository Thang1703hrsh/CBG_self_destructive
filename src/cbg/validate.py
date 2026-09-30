"""Reproducible offline CBG validation; explicit matrices are TOY-ONLY.

Run: python -m src.cbg.validate --output output/cbg_validation.json
This does not download models or certify empirical Llama robustness.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.func import jacrev, jvp, vjp

from src.cbg.functionals import make_safety_functional, make_benign_functional, safe_alignment_loss
from src.cbg.geometry import MetricProduct
from src.cbg.dual import dual_rayleigh_solve
from src.cbg.rayleigh import calibrate_relative_damping, warm_started_rayleigh_ascent
from src.cbg.spectral_grad import (
    centered_spectral_gradient, compute_spectral_center, choose_physical_delta,
    exact_spectral_gradient, make_scalarized_functional, parameter_gradient, perturbed_parameter_tree,
    _gradient_at_perturbation,
)
from src.cbg.tree_ops import (
    select_admissible_parameters, tree_add, tree_cosine_similarity, tree_detach,
    tree_dot, tree_norm, tree_random_like, tree_scale, tree_sub,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 3)
        self.q_proj = nn.Linear(3, 3, bias=False)
        self.v_proj = nn.Linear(3, 3, bias=False)
        self.head = nn.Linear(3, 7, bias=False)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        return (self.head(torch.tanh(self.q_proj(embeddings) + self.v_proj(embeddings))),)


def fixture(device):
    torch.manual_seed(31)
    model = ToyModel().to(device=device, dtype=torch.float64).eval()
    model.requires_grad_(False)
    model.q_proj.requires_grad_(True)
    model.v_proj.requires_grad_(True)
    parameters = select_admissible_parameters(model)
    def batch(rows):
        ids = torch.tensor(rows, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, -2:] = True
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), response_mask=mask)
    harmful = batch([[0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6]])
    safe = batch([[0,1,2,5,6], [1,2,3,6,0], [2,3,4,0,1]])
    benign = batch([[3,2,1,0,6], [4,3,2,1,5], [5,4,3,2,4]])
    return model, parameters, harmful, safe, benign


def legacy_metric(functional, parameters, direction):
    """Pre-optimization one-JVP/one-VJP implementation for benchmarking."""
    _, directional = jvp(functional, (parameters,), (direction,))
    _, pullback = vjp(functional, parameters)
    return tree_detach(pullback(directional / directional.numel())[0])


def legacy_step(fh, fb, parameters, initial, mu, steps):
    # Generic closures intentionally reproduce the old monolithic functional calls.
    h, bfun = lambda p: fh(p), lambda p: fb(p)
    u = initial
    def products(direction):
        gh, gb = legacy_metric(h, parameters, direction), legacy_metric(bfun, parameters, direction)
        bvec = tree_add(gb, direction, alpha=mu)
        denom = tree_dot(direction, bvec)
        return gh, gb, bvec, denom, tree_dot(direction, gh) / denom
    for _ in range(steps):
        gh, gb, bvec, denom, rho = products(u)
        grad = tree_scale(tree_sub(gh, tree_scale(bvec, rho)), 2 / denom)
        candidate = tree_add(u, grad, alpha=.01)
        gb = legacy_metric(bfun, parameters, candidate)
        cost = tree_dot(candidate, gb) + mu * tree_dot(candidate, candidate)
        u = tree_detach(tree_scale(candidate, (cost + 1e-12).rsqrt()))
    products(u)  # The old Rayleigh routine's final diagnostics.
    gh, gb, _, _, _ = products(u)  # Recomputed outer center.
    zh = jvp(h, (parameters,), (u,))[1].detach()
    zb = jvp(bfun, (parameters,), (u,))[1].detach()
    center = compute_spectral_center(fh, fb, parameters, u, mu=mu,
        safety_product=MetricProduct(gh, zh), benign_product=MetricProduct(gb, zb))
    phi = make_scalarized_functional(fh, fb, center)
    delta = choose_physical_delta(parameters, u)
    plus = perturbed_parameter_tree(parameters, u, delta)
    gp = tree_detach(parameter_gradient(phi(plus), plus))
    minus = perturbed_parameter_tree(parameters, u, -delta)
    gm = tree_detach(parameter_gradient(phi(minus), minus))
    return u, tree_scale(tree_sub(gp, gm), 1 / (delta * center.b))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, device, repeats=5):
    function()  # Warm up transforms and kernels; exclude initialization.
    times = []
    peak = None
    for _ in range(repeats):
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = function()
        synchronize(device)
        times.append(1000 * (time.perf_counter() - start))
        if device.type == "cuda":
            peak = max(peak or 0, torch.cuda.max_memory_allocated(device))
        del result
    return dict(median_ms=statistics.median(times), peak_cuda_bytes=peak)


def paired_timing(methods, device, repeats=12):
    """Alternate method order; retain samples rather than claiming stable speedups."""
    samples = {name: [] for name in methods}
    peaks = {name: None for name in methods}
    for method in methods.values():
        method()
    for iteration in range(repeats):
        names = list(methods) if iteration % 2 == 0 else list(reversed(methods))
        for name in names:
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = methods[name]()
            synchronize(device)
            samples[name].append(1000 * (time.perf_counter() - start))
            if device.type == "cuda":
                peaks[name] = max(peaks[name] or 0, torch.cuda.max_memory_allocated(device))
            del result
    return {name: dict(median_ms=statistics.median(times), samples_ms=times,
                       peak_cuda_bytes=peaks[name]) for name, times in samples.items()}


def saved_tensor_accounting(phi, parameters, direction):
    """Peak bytes of saved tensor REFERENCES, not unique storage or peak VRAM."""
    def measure(functional):
        count = dict(live=0, peak=0)
        class Saved:
            def __init__(self, tensor):
                self.tensor = tensor
                self.size = tensor.numel() * tensor.element_size()
                count["live"] += self.size
                count["peak"] = max(count["peak"], count["live"])
            def __del__(self):
                count["live"] -= self.size
        with torch.autograd.graph.saved_tensors_hooks(Saved, lambda box: box.tensor):
            gradient = _gradient_at_perturbation(functional, parameters, direction, 1e-4)
        return gradient, count["peak"]
    monolithic, old_peak = measure(lambda p: phi(p))
    streamed, new_peak = measure(phi)
    relative_error = float(tree_norm(tree_sub(monolithic, streamed)) / tree_norm(monolithic).clamp_min(1e-30))
    assert relative_error < 1e-9 and new_peak < old_peak
    return dict(scope="one Phi endpoint; sum of saved references, NOT peak VRAM or unique storage",
                monolithic_peak_bytes=old_peak, streamed_peak_bytes=new_peak, gradient_relative_error=relative_error)


def run(device):
    model, parameters, harmful, safe, benign = fixture(device)
    fh, fb = make_safety_functional(model, harmful, safe), make_benign_functional(model, benign)
    generator = torch.Generator(device=device).manual_seed(2026)
    initial = tree_random_like(parameters, generator=generator)
    initial = tree_scale(initial, tree_norm(initial).reciprocal())
    mu = float(calibrate_relative_damping(fb, parameters,
        generator=torch.Generator(device=device).manual_seed(7)))
    report = {"scope": "FP64 toy only; no Llama quality or GPU speedup claim",
              "torch": torch.__version__, "device": str(device), "mu": mu,
              "admissible_parameters": sum(p.numel() for p in parameters.values())}
    search = lambda k: warm_started_rayleigh_ascent(fh, fb, parameters,
        mu=mu, inner_steps=k, inner_lr=.01, warm_start=initial)
    results = {k: search(k) for k in (1,2,3,5)}
    report["inner_sweep"] = []
    for k, result in results.items():
        d = result.diagnostics
        report["inner_sweep"].append(dict(K=k, rho=float(d.rho_hat), qh=float(d.qh), qb=float(d.qb),
            b=float(d.qb_damped), b_norm=float(d.b_mu_norm), residual=float(d.eig_residual_norm),
            relative_residual=float(d.relative_residual), converged=d.converged,
            cosine_vs_K5=float(tree_cosine_similarity(result.direction, results[5].direction)),
            history=[float(x) for x in d.rayleigh_values], **timed(lambda: search(k), device)))
    center = results[2].center
    adaptive = warm_started_rayleigh_ascent(fh, fb, parameters, mu=mu, inner_steps=8,
        max_steps=20, max_backtracks=4, inner_lr=.01, warm_start=initial)
    report["candidate_adaptive_search"] = dict(rho=float(adaptive.diagnostics.rho_hat),
        relative_residual=float(adaptive.diagnostics.relative_residual),
        converged=adaptive.diagnostics.converged, accepted_steps=adaptive.diagnostics.inner_steps,
        rejected_trials=adaptive.diagnostics.rejected_steps,
        metric_evaluations=adaptive.diagnostics.metric_evaluations)
    exact = exact_spectral_gradient(make_scalarized_functional(fh, fb, center), parameters, center.direction, center.b)
    snapshot = {name: (p.detach().clone(), p.data_ptr()) for name,p in model.named_parameters()}
    report["fd_sweep"] = []
    for tau in (1e-5, 1e-4, 1e-3):
        fd = centered_spectral_gradient(fh, fb, parameters, center.direction, mu=mu, tau_fd=tau, center=center)
        cosine = float(tree_cosine_similarity(fd.gradient, exact))
        error = float(tree_norm(tree_sub(fd.gradient, exact)) / tree_norm(exact).clamp_min(1e-30))
        restored = all(torch.equal(p, snapshot[n][0]) and p.data_ptr() == snapshot[n][1] for n,p in model.named_parameters())
        report["fd_sweep"].append(dict(tau=tau, delta=float(fd.delta), cosine=cosine, relative_error=error,
            gradient_norm=float(tree_norm(fd.gradient)), cancellation=float(fd.fd_cancellation), restored=restored))
        assert cosine > .95 and error < .01 and restored

    def optimized():
        r = search(2)
        fd = centered_spectral_gradient(fh, fb, parameters, r.direction, mu=mu, center=r.center)
        return r.direction, fd.gradient
    legacy = lambda: legacy_step(fh, fb, parameters, initial, mu, 2)
    old_u, old_g = legacy()
    new_u, new_g = optimized()
    report["equivalence"] = dict(direction_cosine=float(tree_cosine_similarity(old_u, new_u)),
        gradient_relative_error=float(tree_norm(tree_sub(old_g, new_g)) / tree_norm(old_g).clamp_min(1e-30)))
    assert report["equivalence"]["gradient_relative_error"] < 1e-6
    report["benchmark"] = paired_timing({"legacy": legacy, "optimized": optimized}, device)
    report["saved_tensor_accounting"] = saved_tensor_accounting(
        make_scalarized_functional(fh, fb, center), parameters, center.direction,
    )
    for name, method in (("legacy", legacy), ("optimized", optimized)):
        calls = []
        hook = model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            method()
        finally:
            hook.remove()
        report["benchmark"][name]["model_forward_calls"] = len(calls)
    # Explicit matrices are confined to this tiny validation module.
    def jacobian(f):
        jt = jacrev(f)(parameters)
        return torch.cat([jt[n].reshape(3, -1) for n in parameters], dim=1)
    jh, jb = jacobian(fh), jacobian(fb)
    gh, gb = jh.T @ jh / 3, jb.T @ jb / 3
    bm = gb + mu * torch.eye(gh.shape[0], dtype=gh.dtype, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(bm)
    invsqrt = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
    report["exact_toy_rho"] = float(torch.linalg.eigvalsh(invsqrt @ gh @ invsqrt)[-1].detach())
    alignment = parameter_gradient(safe_alignment_loss(model, parameters, safe), parameters)
    preservation_norm, spectral_norm = float(tree_norm(alignment)), float(tree_norm(new_g))
    report["initial_gradient_balance"] = dict(alignment_norm=preservation_norm,
        benign_norm=0., note="reference equals center here; benign KL gradient is zero",
        spectral_norm=spectral_norm, ratio_beta_1e_3=.001 * spectral_norm / preservation_norm,
        beta_for_ratio_025=.25 * preservation_norm / max(spectral_norm, 1e-30))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--v2", action="store_true", help="validate dual solve and token-block benign geometry")
    parser.add_argument("--v5", action="store_true", help="validate direct AD, FD oracle and bounded Ritz")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.v2 and args.v5:
        parser.error("choose only one validation version")
    report = run_v5(torch.device(args.device)) if args.v5 else (run_v2(torch.device(args.device)) if args.v2 else run(torch.device(args.device)))
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def run_v2(device):
    """Small, reproducible diagnostics. No actual-model performance inference."""
    model, parameters, harmful, safe, benign = fixture(device)
    fh = make_safety_functional(model, harmful, safe, chunk_size=1)
    original_fb = make_benign_functional(model, benign, chunk_size=1)
    mu = float(calibrate_relative_damping(original_fb, parameters,
        generator=torch.Generator(device=device).manual_seed(7)))
    report = dict(scope="FP64 toy; fixed mu across metrics, no Llama claims",
                  torch=torch.__version__, device=str(device), mu=mu, variants=[])
    for blocks in (1, 2):
        fb = make_benign_functional(model, benign, chunk_size=1, token_blocks=blocks)
        solve = lambda: dual_rayleigh_solve(fh, fb, parameters, mu=mu, tolerance=1e-8)
        result = solve()
        h_jac, b_jac = jacrev(fh)(parameters), jacrev(fb)(parameters)
        h = torch.cat([v.reshape(3, -1) for v in h_jac.values()], dim=1)
        b = torch.cat([v.reshape(3 * blocks, -1) for v in b_jac.values()], dim=1)
        gh, gb = h.T @ h / 3, b.T @ b / (3 * blocks)
        cholesky = torch.linalg.cholesky(gb + mu * torch.eye(gb.shape[0], device=device, dtype=gb.dtype))
        left = torch.linalg.solve_triangular(cholesky, gh, upper=False)
        sym = torch.linalg.solve_triangular(cholesky, left.T, upper=False).T
        reference = torch.linalg.eigvalsh((sym + sym.T) / 2)[-1]
        exact_gradient = tree_detach(parameter_gradient(reference, parameters))
        fd = centered_spectral_gradient(fh, fb, parameters, result.direction, mu=mu, center=result.center)
        relative_eigen_error = float(((result.center.rho - reference) / reference).abs().detach())
        relative_gradient_error = float(tree_norm(tree_sub(fd.gradient, exact_gradient)) / tree_norm(exact_gradient))
        cosine = float(tree_cosine_similarity(fd.gradient, exact_gradient))
        assert relative_eigen_error < 1e-8 and relative_gradient_error < 1e-5 and cosine > .999
        d = result.diagnostics
        report["variants"].append(dict(token_blocks=blocks, rho=float(d.rho_hat),
            explicit_rho=float(reference.detach()), eigen_relative_error=relative_eigen_error,
            relative_residual=float(d.relative_residual), b_norm=float(d.b_mu_norm),
            damping_fraction=float(d.damping_fraction), eigengap=float(d.eigengap),
            kernel_columns=d.kernel_columns, kernel_projection=float(d.kernel_projection),
            fd_eigen_gradient_relative_error=relative_gradient_error, fd_eigen_gradient_cosine=cosine,
            **timed(solve, device)))
        del h_jac, b_jac, h, b, gh, gb, reference, exact_gradient, fd, result
    report["interpretation"] = (
        "Lower rho across token-block variants is partly a larger denominator by construction; "
        "it is not evidence of better safety. Timings include only the dual solve."
    )
    return report


def run_v5(device):
    """Small numerical V5 qualification; dense matrices are diagnostic only."""
    from src.cbg.spectral_grad import direct_spectral_gradient
    from src.cbg.bounded import bounded_rayleigh_solve, compute_complement_enclosure
    model, parameters, harmful, safe, benign = fixture(device)
    mu = .0006359621327540446
    report = dict(scope="FP64 toy V5 qualification, not a Llama performance result",
                  torch=torch.__version__, device=str(device), mu=mu, variants=[])
    for q in (1, 2):
        fh = make_safety_functional(model, harmful, safe, chunk_size=1)
        fb = make_benign_functional(model, benign, chunk_size=1, token_blocks=q)
        r = dual_rayleigh_solve(fh, fb, parameters, mu=mu, tolerance=1e-8)
        direct = direct_spectral_gradient(fh, fb, parameters, r.direction, mu=mu, center=r.center)
        a = torch.cat([v.reshape(3, -1) for v in jacrev(fh)(parameters).values()], 1) / 3**.5
        c = torch.cat([v.reshape(3*q, -1) for v in jacrev(fb)(parameters).values()], 1) / (3*q)**.5
        chol = torch.linalg.cholesky(c.T@c + mu*torch.eye(c.shape[1], dtype=c.dtype, device=device))
        w = torch.linalg.solve_triangular(chol, a.T, upper=False)
        exact_rho = torch.linalg.eigvalsh(w.T @ w)[-1]
        exact = parameter_gradient(exact_rho, parameters)
        error = float(tree_norm(tree_sub(direct.gradient, exact))/tree_norm(exact))
        assert error < 1e-8
        variant = dict(q=q, rho=float(r.center.rho), b_norm=float(r.center.b.sqrt()),
                       residual=float(r.diagnostics.relative_residual),
                       direct_exact_relative_error=error, gradient_norm=float(tree_norm(direct.gradient)), fd=[])
        for tau in (1e-5, 1e-4, 1e-3):
            fd = centered_spectral_gradient(fh, fb, parameters, r.direction, mu=mu, center=r.center, tau_fd=tau)
            cosine = float(tree_cosine_similarity(direct.gradient, fd.gradient))
            relative = float(tree_norm(tree_sub(fd.gradient, direct.gradient))/tree_norm(direct.gradient))
            assert cosine > .99999 and relative < 1e-4
            variant["fd"].append(dict(tau=tau, cosine=cosine, relative_error=relative,
                                      cancellation=float(fd.fd_cancellation)))
        rr = bounded_rayleigh_solve(fh, fb, parameters, mu=mu,
            generator=torch.Generator(device=device).manual_seed(19))
        assert abs(float(rr.center.rho/r.center.rho)-1) < 1e-3
        variant["bounded"] = dict(rho=float(rr.center.rho), fallback=rr.diagnostics.dual_fallback,
            metric_evaluations=rr.diagnostics.metric_evaluations,
            row_pullbacks=rr.diagnostics.safety_row_pullbacks)
        report["variants"].append(variant)
    p = OrderedDict(w=torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True))
    fh = lambda p: p["w"] * p["w"].new_tensor([2**.5, 200**.5])
    fb = lambda p: p["w"] * 0
    v = OrderedDict(w=p["w"].new_tensor([1., 0.]))
    bound = compute_complement_enclosure(fh, fb, p, [v], mu=1.)
    report["hidden_mode"] = dict(lower=float(bound.lower), upper=float(bound.upper), eta=float(bound.eta))
    return report


if __name__ == "__main__":
    main()
