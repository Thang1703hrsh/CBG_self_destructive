"""Diagnostic: descend log-eta on one frozen S0 geometry batch."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch
import yaml

from src.cbg.spec import (_forward, joint_batch, measure_geometry, measurements,
                          spec_precision, spec_training_gradient)
from src.core.cbg_trainer import CbgArguments
from src.experiments.protocol import digest, file_digest, write_json
from src.experiments.runner import load_model, move_batch
from src.experiments.scenarios import defense_config


def scalar_diagnostics(values):
    return {name: float(value) for name, value in values.items()}


def run(model_snapshot, model_id, model_revision, bank_path, protocol_path,
        choices_path, output_dir, *, steps=20, sign_draws=4):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"fixed-batch diagnostic already exists: {output}")
    if steps < 1 or sign_draws < 1:
        raise ValueError("steps and sign_draws must be positive")
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    choices = yaml.safe_load(Path(choices_path).read_text(encoding="utf-8"))
    if model_id != protocol["models"]["pilot"]:
        raise ValueError("model does not match the prepared pilot")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if bank["pool"] != "defense" or bank["protocol_sha256"] != digest(protocol):
        raise ValueError("geometry bank does not match the prepared defense protocol")
    model, tokenizer = load_model(model_snapshot, "cuda:0")
    if bank["tokenizer_sha256"] != digest(tokenizer.get_vocab()):
        raise ValueError("geometry bank tokenizer differs from the loaded model")
    model.train()
    batch = move_batch(bank["batches"][0], "cuda:0")
    args = CbgArguments(**defense_config(protocol, choices, model=model_id,
                        seed=protocol["seeds"]["defense"][0], output_dir=str(output),
                        pilot=True)["CbgArguments"])
    config = replace(args.spec_config(), alpha=0., drop_aln=True,
                     num_sign_draws=sign_draws)
    seed = protocol["seeds"]["geometry"]
    device = torch.device("cuda:0")
    with spec_precision(model), torch.no_grad():
        joint = joint_batch(batch)
        values = measurements(_forward(model, joint), joint,
                              generator=torch.Generator(device=device).manual_seed(seed))
        fixed_samples = values.samples.detach().clone()
        del values
    if fixed_samples.numel() < 1:
        raise ValueError("fixed geometry batch has no sampled benign tokens")
    sample_digest = digest(fixed_samples.cpu().tolist())
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=protocol["defense"]["learning_rate"],
        weight_decay=protocol["defense"]["weight_decay"])
    max_grad_norm = protocol["defense"]["max_grad_norm"]
    output.mkdir(parents=True)
    write_json(output / "identity.json", dict(
        model=model_id, revision=model_revision, bank_sha256=file_digest(bank_path),
        batch_index=0, sample_sha256=sample_digest, sample_count=fixed_samples.numel(),
        code_sha256=file_digest(__file__), config=asdict(config), steps=steps,
        optimizer="AdamW", learning_rate=protocol["defense"]["learning_rate"],
        max_grad_norm=max_grad_norm, sign_seed_base=seed + 1))
    before = scalar_diagnostics(measure_geometry(model, batch, config=config,
                                                  samples=fixed_samples))
    first = before["log_eta_k"]
    rows = [dict(step=0, **before)]
    print(f"Fixed-batch descent: step=0 log_eta_k={first:.9g}", flush=True)
    with (output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0], allow_nan=False) + "\n")
        stream.flush()
        for step in range(1, steps + 1):
            generator = torch.Generator(device=device).manual_seed(seed + step)
            result = spec_training_gradient(model, batch, config,
                generator=generator, fixed_samples=fixed_samples)
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.grad = result.gradient[name]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                max_grad_norm, error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            timing = float(result.diagnostics["step_time"])
            peak_bytes = int(result.diagnostics["peak_memory"])
            del result
            measured = scalar_diagnostics(measure_geometry(model, batch, config=config,
                                                            samples=fixed_samples))
            row = dict(step=step, **measured, grad_norm=grad_norm,
                       gradient_time_seconds=timing, gradient_peak_bytes=peak_bytes)
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"nonfinite diagnostic at step {step}")
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(f"Fixed-batch descent: step={step}/{steps} "
                  f"log_eta_k={row['log_eta_k']:.9g} "
                  f"delta={row['log_eta_k'] - first:+.9g} "
                  f"grad_norm={grad_norm:.6g}", flush=True)
    summary = dict(status="decreased" if rows[-1]["log_eta_k"] < first else "not_decreased",
        initial_log_eta_k=first, final_log_eta_k=rows[-1]["log_eta_k"],
        change=rows[-1]["log_eta_k"] - first,
        steps_decreased=sum(rows[i]["log_eta_k"] < rows[i-1]["log_eta_k"]
                            for i in range(1, len(rows))),
        min_log_eta_k=min(row["log_eta_k"] for row in rows),
        max_peak_memory_gib=max(row["gradient_peak_bytes"] for row in rows[1:]) / 2**30)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model-snapshot", "model-id", "model-revision", "bank", "protocol",
                 "choices", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sign-draws", type=int, default=4)
    options = vars(parser.parse_args())
    run(options["model_snapshot"], options["model_id"], options["model_revision"],
        options["bank"], options["protocol"], options["choices"], options["output"],
        steps=options["steps"], sign_draws=options["sign_draws"])
