"""Bounded, real-model L1--L6 correctness checks for CBG v2 S0.1.

No optimizer is constructed. Large independent Jacobian rows live on the CPU;
only short batches and one row gradient at a time are differentiated on device.
These tests do not substitute for the separately supplied cbg_reference.py.
"""
from collections import OrderedDict
from dataclasses import asdict
import copy
import gc
import math
from pathlib import Path
import time

import torch

from src.cbg.spec import (SpecConfig, _center, _forward, joint_batch, measurements,
                          spec_precision, spec_training_gradient)
from src.cbg.spec_hooks import spec_linear_modules
from src.cbg.spec_spectral import eigen_sign_scalarization
from src.experiments.geometry import radius_curve
from src.experiments.protocol import write_json


def synthetic_batch(vocab_size, options, device, seed):
    """Deterministic token IDs; semantics are irrelevant to derivative equality."""
    m, n = options["safety_pairs"], options["benign_sequences"]
    length, response = options["sequence_length"], options["response_tokens"]
    if min(m, n) < 1 or not 1 <= response < length or length > 32:
        raise ValueError("S0.1 requires nonempty batches and 1 <= response < length <= 32")
    rng = torch.Generator().manual_seed(seed)
    def group(count):
        ids = torch.randint(3, min(vocab_size, 256), (count, length), generator=rng).to(device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, -response:] = True
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), response_mask=mask)
    result = dict(harmful=group(m), safe=group(m), benign=group(n))
    result["safe"]["input_ids"][:, :-response] = result["harmful"]["input_ids"][:, :-response]
    return result


def relative_error(actual, expected):
    a, b = actual.detach().double().cpu(), expected.detach().double().cpu()
    return float((a-b).norm()/b.norm().clamp_min(1e-30))


def assert_error(value, threshold, label):
    if not math.isfinite(value) or value > threshold:
        raise AssertionError(f"{label}: relative error {value:.6g} exceeds {threshold:g}")


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model_checks(model, options, output):
    """Run all L checks, writing a receipt after each check and on failure.

    The caller supplies model provenance/reference-test status. This receipt
    certifies only L1--L6, never the whole S0.1 gate or a production radius.
    """
    output = Path(output)
    device = next(model.parameters()).device
    count = sum(p.numel() for p in model.parameters())
    if count > 1_500_000_000:
        raise ValueError("S0.1 runner is capped at the Llama-3.2-1B class (1.5B parameters)")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("L1--L6 must use FP32 master weights")
    model.requires_grad_(True)
    model.eval()
    model.config.use_cache = False
    config = SpecConfig(**options["method"])
    if config.perturb_precision != "fp32" or config.capture_dtype != "model":
        raise ValueError("S0.1 baseline requires FP32 compute and uncompressed captures")
    batch = synthetic_batch(model.config.vocab_size, options["batch"], device, options["seed"])
    joint = joint_batch(batch)
    modules = spec_linear_modules(model, config.block_path)
    parameters = OrderedDict(model.named_parameters())
    sampled = torch.randint(0, model.config.vocab_size,
        (options["batch"]["benign_sequences"]*options["batch"]["response_tokens"],),
        generator=torch.Generator().manual_seed(options["seed"]+1)).to(device)
    limits = options["checks"]
    report = dict(status="running", scenario="S0.1", scope="L1-L6", method_revision="cbg_v2",
                  model_parameters=count, device=str(device), config=asdict(config),
                  batch=options["batch"], thresholds=limits, seed=options["seed"], checks={})
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
    write_json(output, report)

    def check(name, action):
        _release()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        print(f"S0.1 {name}: starting", flush=True)
        try:
            result = action()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report["checks"][name] = dict(status="passed", seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                **result)
        except Exception as error:
            report["status"] = "failed"
            report["checks"][name] = dict(status="failed", error=f"{type(error).__name__}: {error}",
                                          seconds=time.perf_counter()-started)
            write_json(output, report)
            raise
        write_json(output, report)
        print(f"S0.1 {name}: {report['checks'][name]}", flush=True)

    state = {}
    def gram_and_rows():
        with spec_precision(model):
            q, _ = _center(model, joint, config, modules, samples=sampled, materialize=False)
            # Independent ordinary per-example autograd, not hook reconstruction.
            rows = {name: [] for name in modules}
            for e in range(joint.m+joint.n):
                score = measurements(_forward(model, joint), joint, samples=sampled).values[e]
                grads = torch.autograd.grad(score, tuple(module.weight for module in modules.values()))
                for name, gradient in zip(modules, grads):
                    rows[name].append(gradient.detach().cpu())
                del score, grads, gradient
            expected = torch.zeros_like(q.kernel, device="cpu")
            # Chunk even the independent CPU oracle; never allocate E x |U| in
            # a single tensor or convert the whole parameter Jacobian to FP64.
            for name, module in modules.items():
                weight = module.weight.detach().cpu()
                for start in range(0, weight.shape[0], 32):
                    C = torch.stack([r[start:start+32].flatten() for r in rows[name]]).double()
                    lam = (weight[start:start+32].square()+config.xi**2).flatten().double()
                    expected += (C*lam)@C.T
            error = relative_error(q.kernel, expected)
            assert_error(error, limits["gram_relative_error"], "L1 Gram")
            state.update(rows=rows, kernel=q.kernel.detach().cpu())
            return dict(relative_error=error, m=joint.m, n=joint.n, mu=float(q.mu))
    check("L1", gram_and_rows)

    def directions():
        with spec_precision(model):
            q, _, captures = _center(model, joint, config, modules, samples=sampled, return_capture=True)
            b, _ = eigen_sign_scalarization(q, torch.ones(joint.m+joint.n, device=device))
            captures.materialize(b, q.A)
        b, G = b.cpu(), q.G.cpu()
        errors, norms = [0., 0.], [0., 0.]
        actual_metric_norm = 0.
        for name, module in modules.items():
            weight = module.weight.detach().cpu()
            for start in range(0, weight.shape[0], 32):
                C = torch.stack([r[start:start+32].flatten() for r in state["rows"][name]]).double()
                theta = weight[start:start+32].double()
                lam = theta.square()+config.xi**2
                expected_w = (b@C).reshape_as(theta)*lam
                expected_dl = 2*theta*(C*(G@C)).sum(0).reshape_as(theta)
                for i, (actual, expected) in enumerate(((captures.w[name][start:start+32], expected_w),
                                                      (captures.d_lambda[name][start:start+32], expected_dl))):
                    value = actual.detach().double().cpu()
                    errors[i] += float((value-expected).square().sum())
                    norms[i] += float(expected.square().sum())
                actual_metric_norm += float((captures.w[name][start:start+32].detach().double().cpu().square()/lam).sum())
        w_error, dl_error = [(e/max(n, 1e-60))**.5 for e, n in zip(errors, norms)]
        assert_error(w_error, limits["direction_relative_error"], "L2 direction")
        assert_error(dl_error, limits["correction_relative_error"], "L2 metric correction")
        expected_norm = float(b@q.kernel.cpu()@b)
        norm_error = abs(actual_metric_norm-expected_norm)/max(abs(expected_norm), 1e-30)
        assert_error(norm_error, limits["direction_relative_error"], "L2 radius identity")
        del state["rows"]
        return dict(direction_relative_error=w_error, correction_relative_error=dl_error,
                    radius_identity_relative_error=norm_error)
    check("L2", directions)

    def hvp():
        result = radius_curve(model, batch, config, limits["hvp_radii"], seed=options["seed"],
            perturb_precision="fp32", microbatches=limits["hvp_microbatches"], output_deltas=())
        write_json(output.with_name("L3_radius.json"), result)
        best = min(row["relative_error"] for row in result["curve"])
        assert_error(best, limits["hvp_relative_error"], "L3 finite difference versus exact HVP")
        return dict(best_relative_error=best, curve=result["curve"], production_radius_selected=False)
    check("L3", hvp)

    def base_gradient():
        from dataclasses import replace
        benign = batch["benign"]
        with spec_precision(model), torch.no_grad():
            logits = model(input_ids=benign["input_ids"], attention_mask=benign["attention_mask"]).logits
            reference = logits[:, :-1][benign["response_mask"][:, 1:]].detach().cpu()
            del logits
        result = spec_training_gradient(model, batch, replace(config, beta=0.),
                                         reference_logits=reference, fixed_samples=sampled)
        expected = {name: g.detach().cpu() for name, g in result.gradient.items()}
        del result
        with spec_precision(model):
            value = measurements(_forward(model, joint), joint, samples=sampled, reference_logits=reference)
            loss = value.alignment+config.alpha*value.benign_kl
            gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        numerator, denominator = 0., 0.
        for name, grad in zip(parameters, gradients):
            # Chunk to avoid temporary FP64 copies of a large embedding/head.
            left, right = grad.detach().flatten(), expected[name].flatten()
            for start in range(0, left.numel(), 1_000_000):
                a, b = left[start:start+1_000_000].double().cpu(), right[start:start+1_000_000].double()
                numerator += float((a-b).square().sum())
                denominator += float(b.square().sum())
        error = (numerator/max(denominator, 1e-60))**.5
        assert_error(error, limits["base_gradient_relative_error"], "L4 beta-zero gradient")
        return dict(relative_error=error)
    check("L4", base_gradient)

    def padding():
        padded = copy.deepcopy(batch)
        for group in padded.values():
            for field, value in group.items():
                group[field] = torch.nn.functional.pad(value, (0, options["batch"]["padding_tokens"]), value=0)
        with spec_precision(model):
            q, _ = _center(model, joint_batch(padded), config, modules, samples=sampled, materialize=False)
        error = relative_error(q.kernel, state["kernel"])
        assert_error(error, limits["padding_relative_error"], "L5 padding")
        return dict(relative_error=error)
    check("L5", padding)

    def checkpointing():
        original = getattr(model, "is_gradient_checkpointing", False)
        mode = model.training
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            with spec_precision(model):
                q, _ = _center(model, joint, config, modules, samples=sampled, materialize=False)
            error = relative_error(q.kernel, state["kernel"])
            assert_error(error, limits["checkpoint_relative_error"], "L6 checkpoint configuration")
        finally:
            if not original:
                model.gradient_checkpointing_disable()
            model.train(mode)
        return dict(relative_error=error, policy="center eval mode disables checkpoint replay",
                    checkpoint_memory_savings_tested=False)
    check("L6", checkpointing)
    report["status"] = "passed"
    report["production_radius_selected"] = False
    write_json(output, report)
    return report
