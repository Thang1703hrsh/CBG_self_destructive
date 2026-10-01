"""Five-update CBG diagnostic with fixed-batch holdouts and gradient attribution."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch
import yaml

from src.cbg.functionals import _extract_logits, select_response_logits
from src.cbg.spec import _center, joint_batch, spec_precision, spec_training_gradient
from src.cbg.spec_hooks import spec_linear_modules
from src.core.cbg_trainer import CbgArguments
from src.experiments.protocol import digest, file_digest, write_json
from src.experiments.runner import load_model, move_batch
from src.experiments.scenarios import defense_config


def effective_rank(block):
    """Stable participation ratio tr(K)^2 / tr(K^2) for a Gram block."""
    block = block.detach().double()
    square_trace = block.square().sum()
    return float(block.trace().square() / square_trace) if square_trace > 0 else 0.0


def _response_loss(model, group):
    logits = _extract_logits(model(input_ids=group["input_ids"],
                                    attention_mask=group["attention_mask"]))
    selected = select_response_logits(logits, group)
    mask = group["response_mask"][:, 1:].bool() & group["attention_mask"][:, 1:].bool()
    targets = group["input_ids"][:, 1:][mask]
    return -selected.log_softmax(-1).gather(1, targets[:, None]).mean()


def _benign_logits(model, group):
    logits = _extract_logits(model(input_ids=group["input_ids"],
                                    attention_mask=group["attention_mask"]))
    return select_response_logits(logits, group)


def _benign_kl(model, group, reference_logits):
    current = _benign_logits(model, group).log_softmax(-1)
    reference = reference_logits.to(current).log_softmax(-1)
    return (reference.exp() * (reference - current)).sum(-1).mean()


def _norm_and_dot(grads, geometry, names):
    squared = torch.zeros((), dtype=torch.float64, device=next(iter(geometry.values())).device)
    dot = torch.zeros_like(squared)
    for grad, name in zip(grads, names):
        if grad is not None:
            squared += grad.detach().double().square().sum()
            dot += (grad.detach().double() * geometry[name].detach().double()).sum()
    return float(squared.sqrt()), float(dot)


def _gradient_attribution(model, batch, reference_logits, geometry):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names = [name for name, _ in parameters]
    leaves = [p for _, p in parameters]
    geometry_norm = math.sqrt(sum(float(g.detach().double().square().sum())
                                  for g in geometry.values()))
    result = {"grad_norm_beta_cbg": geometry_norm}
    with spec_precision(model):
        alignment = _response_loss(model, batch["safe"])
        grads = torch.autograd.grad(alignment, leaves, allow_unused=True)
        norm, dot = _norm_and_dot(grads, geometry, names)
        result.update(grad_norm_aln=norm,
                      cos_aln_beta_cbg=dot / max(norm * geometry_norm, 1e-30))
        del alignment, grads
        benign = _benign_kl(model, batch["benign"], reference_logits)
        grads = torch.autograd.grad(benign, leaves, allow_unused=True)
        norm, dot = _norm_and_dot(grads, geometry, names)
        result.update(grad_norm_bgn=norm,
                      cos_bgn_beta_cbg=dot / max(norm * geometry_norm, 1e-30))
        del benign, grads
    return result


def _measure(model, batch, config, samples, reference_logits, *, seed):
    with spec_precision(model):
        joint = joint_batch(batch)
        spectrum, used_samples = _center(model, joint, config,
            spec_linear_modules(model, config.block_path),
            generator=torch.Generator(device=joint.input_ids.device).manual_seed(seed),
            samples=samples, materialize=False)
        diagnostics = {name: float(value) for name, value in spectrum.diagnostics().items()}
        kernel = spectrum.kernel
        diagnostics["effective_rank_K_hh"] = effective_rank(kernel[:joint.m, :joint.m])
        diagnostics["effective_rank_K_bb"] = effective_rank(kernel[joint.m:, joint.m:])
        diagnostics["top_nu_share_K_hh"] = float(spectrum.nu[0] / kernel[:joint.m, :joint.m].trace())
        del spectrum, kernel
    with spec_precision(model), torch.no_grad():
        diagnostics["L_aln"] = float(_response_loss(model, batch["safe"]))
        diagnostics["L_bgn"] = float(_benign_kl(model, batch["benign"], reference_logits))
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise FloatingPointError("nonfinite holdout diagnostic")
    return diagnostics, used_samples.detach()


def run(model_snapshot, model_id, model_revision, bank_path, protocol_path,
        choices_path, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"diagnostic already exists: {output}")
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    choices = yaml.safe_load(Path(choices_path).read_text(encoding="utf-8"))
    if model_id != protocol["models"]["pilot"]:
        raise ValueError("model does not match the prepared pilot")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if (bank["pool"] != "defense" or bank["protocol_sha256"] != digest(protocol)
            or len(bank["batches"]) < 4):
        raise ValueError("prepared defense bank does not match or lacks holdouts")
    model, tokenizer = load_model(model_snapshot, "cuda:0")
    if bank["tokenizer_sha256"] != digest(tokenizer.get_vocab()):
        raise ValueError("bank tokenizer differs from model")
    model.train()
    batches = [move_batch(bank["batches"][i], "cuda:0") for i in range(4)]
    args = CbgArguments(**defense_config(protocol, choices, model=model_id,
        seed=protocol["seeds"]["defense"][0], output_dir=str(output),
        pilot=True)["CbgArguments"])
    config = replace(args.spec_config(), alpha=0., drop_aln=True, num_sign_draws=4)
    seed = protocol["seeds"]["geometry"]
    references = []
    for batch in batches:
        with spec_precision(model), torch.no_grad():
            references.append(_benign_logits(model, batch["benign"]).detach().cpu())
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=protocol["defense"]["learning_rate"],
        weight_decay=protocol["defense"]["weight_decay"])
    max_grad_norm = protocol["defense"]["max_grad_norm"]
    output.mkdir(parents=True)
    samples = [None] * 4
    rows = []
    def record(step, attribution=None, train_norm=None, peak_bytes=None):
        measured = []
        for i, batch in enumerate(batches):
            diag, samples[i] = _measure(model, batch, config, samples[i], references[i],
                                        seed=seed + i)
            measured.append(diag)
            print(f"five-step: step={step}/5 batch={i} log_eta_k={diag['log_eta_k']:.8g} "
                  f"rank_bb={diag['effective_rank_K_bb']:.5g} "
                  f"L_bgn={diag['L_bgn']:.6g}", flush=True)
        row = dict(step=step, train=measured[0], holdouts=measured[1:],
                   gradient_attribution=attribution,
                   optimizer_grad_norm_preclip=train_norm,
                   gradient_peak_memory_bytes=peak_bytes)
        rows.append(row)
        with (output / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row
    record(0)
    write_json(output / "identity.json", dict(model=model_id, revision=model_revision,
        bank_sha256=file_digest(bank_path), code_sha256=file_digest(__file__),
        batch_indices=[0, 1, 2, 3],
        sample_sha256=[digest(s.cpu().tolist()) for s in samples],
        reference="initial model selected benign logits, per batch",
        config=asdict(config), steps=5, optimizer="AdamW",
        learning_rate=protocol["defense"]["learning_rate"],
        max_grad_norm=max_grad_norm))
    for step in range(1, 6):
        generator = torch.Generator(device="cuda:0").manual_seed(seed + step)
        result = spec_training_gradient(model, batches[0], config,
                                        generator=generator, fixed_samples=samples[0])
        attribution = _gradient_attribution(model, batches[0], references[0],
                                            result.gradient)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = result.gradient[name]
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                     error_if_nonfinite=True))
        peak = int(result.diagnostics["peak_memory"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del result
        record(step, attribution=attribution, train_norm=norm, peak_bytes=peak)
    before, after = rows[0], rows[-1]
    summary = dict(status="complete", steps=5,
        train_change_log_eta_k=after["train"]["log_eta_k"] - before["train"]["log_eta_k"],
        holdout_changes_log_eta_k=[a["log_eta_k"] - b["log_eta_k"]
                                  for a, b in zip(after["holdouts"], before["holdouts"])],
        train_rank_bb_start=before["train"]["effective_rank_K_bb"],
        train_rank_bb_end=after["train"]["effective_rank_K_bb"],
        max_peak_memory_gib=max(row["gradient_peak_memory_bytes"] for row in rows[1:]) / 2**30)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
