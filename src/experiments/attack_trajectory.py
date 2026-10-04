"""Attack-trajectory diagnostics of CBG's mechanism (one model, its attack snapshots).

Inputs: a defended (or base) model and the snapshots of attacks run from it with
``attack_eval --save-at-steps`` (full fine-tuning: model directories ``step-<n>``; LoRA: adapter
directories ``step-<n>``). For the starting point and every snapshot it measures:

  geometry  the held-out geometry on D_geo (log eta_k, sigma_1:k, nu_1:k, tr K_bb, effective ranks), as
            ``train_v2 --stage geometry``: does the trained barrier survive the attack?
  cosine    on each D_geo batch, the cosine in the relative metric Lambda = diag(1 / (theta_s^2 + xi^2))
            between the attack displacement u = theta_t - theta_s and the top safety direction
            v = Lambda^{-1} C_h^T phi of the starting model, where phi is the top eigenvector of K_hh
            (``Prop. eta (iii)``), oriented so that sum(phi) > 0. The safety margin F_h is
            mean log p(y^h) - mean log p(y^s) (src/cbg/spec.py), so a positive cosine means the attack moves
            along the top safety direction toward the harmful responses; cos^2 is the fraction of the
            attack's (metric) displacement that lies on it.

theta_s is the starting model rounded to bf16: the bf16 attack starts from there, so u is the exact attack
displacement. Only the Linear weights of the transformer blocks (the space U of the paper) enter the cosine.

Usage:
  python -m src.experiments.attack_trajectory --config <cbg yaml> --data <data dir> --model <start model>
         --attack lora=<attacked-1e-4 dir> --attack mixed10=<attacked-5e-05 dir> --output <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import torch

from src.cbg.spec import joint_batch, measurements, spec_precision
from src.cbg.spec_hooks import spec_linear_modules
from src.cbg.step_v2 import _autocast, _logits, center_pass, make_chunks, make_totals
from src.cbg.train_v2 import (datasets, geometry_batches, geometry_eval, gpu_devices, load_config,
                              load_split_model, step_config, to_device)
from src.experiments.protocol import write_json

STEP = re.compile(r"step-(\d+)$")


def snapshots(directory):
    """(step, path) of every step-<n> snapshot under ``directory``, sorted by step."""
    found = [(int(STEP.search(p.name).group(1)), p) for p in Path(directory).iterdir()
             if p.is_dir() and STEP.search(p.name)]
    if not found:
        raise FileNotFoundError(f"no step-<n> snapshots under {directory}")
    return sorted(found)


def _read_safetensors(path):
    from safetensors.torch import load_file
    tensors = {}
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {path}")
    for file in files:
        tensors.update(load_file(file))
    return tensors


def lora_deltas(path, names):
    """Exact fp32 weight updates scale * B @ A of a PEFT LoRA adapter, keyed by U weight name."""
    config = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    scale = config["lora_alpha"] / config["r"]
    tensors = _read_safetensors(path)
    deltas = {}
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        module = key.split(".lora_A.")[0]
        name = module[module.index("model.layers."):] + ".weight"
        if name not in names:
            raise KeyError(f"LoRA target {name} is not in U")
        deltas[name] = scale * (b.float() @ a.float())
    if not deltas:
        raise ValueError(f"no LoRA weights in {path}")
    return deltas


class Snapshot:
    """One attack snapshot: its U displacement (resident on the weights' devices) and how to load it."""

    def __init__(self, attack, step, path, kind):
        self.attack, self.step, self.path, self.kind = attack, step, Path(path), kind
        self.u = {}

    def load_displacement(self, params, start):
        if self.kind == "lora":
            for name, delta in lora_deltas(self.path, params).items():
                self.u[name] = delta.to(params[name].device)
        else:
            tensors = _read_safetensors(self.path)
            for name, param in params.items():
                self.u[name] = (tensors[name].float().to(param.device) - start[name].to(param.device)).bfloat16()

    def apply(self, model, params, start_all):
        """Set the model to this snapshot (fp32)."""
        with torch.no_grad():
            if self.kind == "lora":
                for name, delta in self.u.items():
                    params[name].copy_(start_all[name].to(params[name].device) + delta.float())
            else:
                tensors = _read_safetensors(self.path)
                for name, param in model.named_parameters():
                    param.copy_(tensors[name].to(param.device, torch.float32))


def restore(model, start_all):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(start_all[name].to(param.device))


def top_safety_gradient(model, batch, cfg, modules, generator):
    """phi (top eigenvector of K_hh, sum > 0), nu_1, its trace share, and grad_theta sum_i phi_i F_h,i in .grad."""
    joint = joint_batch(batch)
    chunks = make_chunks(joint, cfg.microbatch_tokens)
    totals = make_totals(joint)
    with spec_precision(model, cfg.perturb_precision):
        hooks, _ = center_pass(model, joint, chunks, totals, cfg, modules, sample_generator=generator)
        kernel = hooks.K.double()
        hooks.captures.clear()
    hh = kernel[:joint.m, :joint.m]
    values, vectors = torch.linalg.eigh(hh)
    phi = vectors[:, -1]
    if float(phi.sum()) < 0:
        phi = -phi
    model.zero_grad(set_to_none=True)
    device_type = joint.input_ids.device.type
    for chunk in chunks:
        if chunk.batch.n:   # benign micro-batches carry no safety margin
            continue
        with _autocast(device_type, cfg.center_precision):
            logits = _logits(model, chunk.batch)
        result = measurements(logits, chunk.batch, fisher=cfg.fisher, logprob_chunk_size=cfg.logprob_chunk_size,
                              totals=totals)
        weights = phi.to(result.values.device, result.values.dtype)[chunk.examples.to(result.values.device)]
        (result.values * weights).sum().backward()
        del logits, result
    return dict(phi=phi.tolist(), nu_1=float(values[-1]), top_share=float(values[-1] / hh.trace()))


def run(config_path, data_dir, model_path, attacks, output):
    from transformers import AutoTokenizer
    config = load_config(config_path)
    cfg = step_config(config)
    devices = gpu_devices(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = geometry_batches(datasets(config, data_dir, tokenizer), config)
    xi = float(config["method"]["xi"])

    model = load_split_model(model_path, devices, trainable=False)
    with torch.no_grad():   # theta_s: the bf16 point the attacks start from
        for param in model.parameters():
            param.copy_(param.bfloat16().float())
    modules = spec_linear_modules(model, cfg.block_path)
    params = {name: module.weight for name, module in modules.items()}
    start_all = {name: param.detach().to("cpu", copy=True) for name, param in model.named_parameters()}

    snaps = []
    for attack, directory in attacks:
        for step, path in snapshots(directory):
            kind = "lora" if (path / "adapter_config.json").exists() else "full"
            snap = Snapshot(attack, step, path, kind)
            snap.load_displacement(params, {n: start_all[n] for n in params})
            snaps.append(snap)
            print(f"loaded {attack} step {step} ({kind})", flush=True)

    # Geometry at the start and at every snapshot.
    rows = []
    summary, _ = geometry_eval(model, batches, cfg, config, devices)
    rows.append(dict(attack="start", step=0, **summary))
    print(json.dumps(rows[-1]), flush=True)
    for snap in snaps:
        snap.apply(model, params, start_all)
        summary, _ = geometry_eval(model, batches, cfg, config, devices)
        rows.append(dict(attack=snap.attack, step=snap.step, kind=snap.kind, **summary))
        print(json.dumps(rows[-1]), flush=True)
        restore(model, start_all)

    # Cosine with the top safety direction of the starting model, per D_geo batch.
    for param in params.values():
        param.requires_grad_(True)
    directions, cosines = [], {(s.attack, s.step): [] for s in snaps}
    for index, batch in enumerate(batches):
        generator = torch.Generator(device=devices[-1]).manual_seed(config["seeds"]["d_geo_y_hat"] + index)
        info = top_safety_gradient(model, to_device(batch, devices[0]), cfg, modules, generator)
        g_norm = 0.
        dots = {key: 0. for key in cosines}
        u_norms = {key: 0. for key in cosines}
        with torch.no_grad():
            for name, param in params.items():
                g = param.grad.float()
                inv_metric = param.float().square() + xi ** 2      # Lambda^{-1}
                g_norm += float((g.square() * inv_metric).sum(dtype=torch.float64))
                for snap in snaps:
                    u = snap.u.get(name)
                    if u is None:
                        continue
                    u = u.float()
                    key = (snap.attack, snap.step)
                    dots[key] += float((u * g).sum(dtype=torch.float64))
                    u_norms[key] += float((u.square() / inv_metric).sum(dtype=torch.float64))
        model.zero_grad(set_to_none=True)
        for key in cosines:
            # A snapshot taken before the first nonzero learning rate (warmup) has u = 0: the cosine is undefined.
            denominator = (u_norms[key] * g_norm) ** .5
            cosines[key].append(dots[key] / denominator if denominator > 0 else None)
        directions.append(dict(batch=index, nu_1=info["nu_1"], top_share=info["top_share"]))
        print(json.dumps(dict(batch=index, top_share=info["top_share"],
                              cos={f"{a}@{s}": c[-1] for (a, s), c in cosines.items()})), flush=True)

    trajectory = []
    for snap in snaps:
        values = cosines[(snap.attack, snap.step)]
        defined = [v for v in values if v is not None]
        geo = next(r for r in rows if r["attack"] == snap.attack and r["step"] == snap.step)
        trajectory.append(dict(attack=snap.attack, step=snap.step, kind=snap.kind,
                               cos_top_safety_mean=statistics.mean(defined) if defined else None,
                               cos_top_safety_per_batch=values,
                               cos2_mean=statistics.mean(v * v for v in defined) if defined else None,
                               log_eta_k=geo["geo/log_eta_k"]))
    result = dict(model=str(model_path), config=str(config_path), xi=xi, start="model rounded to bf16",
                  metric="relative, Lambda^{-1} = theta_s^2 + xi^2, U = Linear weights in the blocks",
                  geometry=rows, top_direction=directions, trajectory=trajectory)
    write_json(Path(output), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="the model the attacks started from")
    parser.add_argument("--attack", action="append", required=True,
                        help="name=<directory with step-<n> snapshots>; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    attacks = [tuple(item.split("=", 1)) for item in args.attack]
    run(args.config, args.data, args.model, attacks, args.output)


if __name__ == "__main__":
    main()
