"""Upload a finished CBG v2 run to the Hugging Face Hub (one atomic commit).

Contents of the commit:
  final/*                    the trained model + tokenizer (as saved by train_v2), optionally cast
  training/<records>         config.yaml, identity.json, metrics.json, train_log.jsonl
  README.md                  model card (base model, method settings, final geometry, provenance)

The repository is created private unless ``private=False``. The token comes
from HF_TOKEN (needs write access). A record of the upload is written to
<run_dir>/hf_upload.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from src.experiments.protocol import write_json

REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
TRAINING_RECORDS = ("config.yaml", "identity.json", "metrics.json", "train_log.jsonl")
DTYPES = {"float32": "F32", "bfloat16": "BF16", "float16": "F16"}


def check_repo_id(repo_id):
    if not REPO_ID.fullmatch(repo_id or ""):
        raise ValueError(f"invalid Hugging Face repo id {repo_id!r}; expected <user or org>/<name>")
    return repo_id


def cast_model(final, target, dtype):
    """Copy ``final`` to ``target`` with floating safetensors weights cast to ``dtype``."""
    import torch
    from safetensors.torch import load_file, save_file
    torch_dtype = getattr(torch, dtype)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in sorted(Path(final).iterdir()):
        if path.suffix == ".safetensors":
            tensors = {name: value.to(torch_dtype) if value.is_floating_point() else value
                       for name, value in load_file(path).items()}
            total += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, target / path.name, metadata={"format": "pt"})
        elif path.is_file():
            shutil.copy2(path, target / path.name)
    config = target / "config.json"
    if config.exists():
        values = json.loads(config.read_text(encoding="utf-8"))
        values["torch_dtype"] = dtype
        config.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    index = target / "model.safetensors.index.json"
    if index.exists():
        values = json.loads(index.read_text(encoding="utf-8"))
        values.setdefault("metadata", {})["total_size"] = total
        index.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return target


def _last_geometry(log_path):
    if not log_path.exists():
        return {}
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    geo = [r for r in rows if "geo/log_eta_k" in r]
    if not geo:
        return {}
    first, last = geo[0], geo[-1]
    return {"geo/log_eta_k (step 0)": first["geo/log_eta_k"], f"geo/log_eta_k (step {last['step']})":
            last["geo/log_eta_k"], "delta log eta_k": last["geo/log_eta_k"] - first["geo/log_eta_k"]}


def model_card(run_dir, repo_id, *, dtype, base_model, base_revision=None):
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) \
        if (run_dir / "config.yaml").exists() else {}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    method, opt, batch = config.get("method", {}), config.get("optimizer", {}), config.get("batch", {})
    settings = {
        "steps": metrics.get("steps"), "lr": opt.get("lr"), "scheduler": opt.get("scheduler"),
        "batch m / n": f"{batch.get('m')} / {batch.get('n')}", "alpha": method.get("alpha"),
        "beta": method.get("beta"), "k": method.get("k"), "radius r": method.get("radius"),
        "lazy_period": method.get("lazy_period", 1), "lazy_mode": method.get("lazy_mode", "rescale"),
        "num_sign_draws": method.get("num_sign_draws"), "geometry_clip_factor": method.get("geometry_clip_factor"),
        "max length (safety / benign)": "{} / {}".format(*(config.get("lengths", {}).get(k) for k in
                                                           ("safety", "benign"))),
        "weights dtype": dtype,
    }
    identity = json.loads((run_dir / "identity.json").read_text(encoding="utf-8")) \
        if (run_dir / "identity.json").exists() else {}
    provenance = {"base model": base_model + (f" @ {base_revision}" if base_revision else ""),
                  "config sha256": identity.get("config_sha256"),
                  "data manifest sha256": identity.get("data_manifest_sha256"),
                  "torch": identity.get("torch"), "GPUs": ", ".join(identity.get("gpus", []))}
    table = lambda values: "\n".join(["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in values.items()
                                                                 if v is not None])
    geometry = _last_geometry(run_dir / "train_log.jsonl")
    header = yaml.safe_dump(dict(base_model=base_model, license="llama2", library_name="transformers",
                                 pipeline_tag="text-generation", tags=["cbg", "safety", "tamper-resistance"]),
                            sort_keys=False)
    return "\n".join([
        "---", header.strip(), "---", "",
        f"# {repo_id.split('/')[-1]}", "",
        f"Full-parameter CBG v2 defense of `{base_model}`, trained with `src/cbg/train_v2.py` on the SEAM "
        "main-run data recipe (RepNoise BeaverTails refusals, 4,000 defense rows; Alpaca, 4,000 benign rows).", "",
        "Prompt format: `Question: {prompt}\\nAnswer:` (no chat template), as in training.", "",
        "## Training settings", "", table(settings), "",
        *(["## Geometry (D_geo)", "", table(geometry), ""] if geometry else []),
        "## Provenance", "", table(provenance), "",
        "Training records (config, identity, per-step log, metrics) are in `training/`.", "",
        "## License", "",
        "A derivative of Llama 2, distributed under the Llama 2 Community License Agreement; "
        "use is subject to its terms and Acceptable Use Policy.", ""])


def upload_run(run_dir, repo_id, *, private=True, dtype="float32", base_model=None, base_revision=None,
               api=None, token=None, staging_dir=None, commit_message=None):
    """Upload <run_dir>/final plus training records and a model card; returns the upload record."""
    from huggingface_hub import CommitOperationAdd, HfApi
    run_dir = Path(run_dir)
    check_repo_id(repo_id)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    metrics_path, final = run_dir / "metrics.json", run_dir / "final"
    if not metrics_path.exists() or json.loads(metrics_path.read_text(encoding="utf-8")).get("status") != "complete" \
            or not final.is_dir():
        raise FileNotFoundError(f"{run_dir} has no completed training run (metrics.json + final/)")
    if base_model is None:
        base_model = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["model"]
    api = api or HfApi(token=token)
    repo_url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_dir) as tmp:
        tmp = Path(tmp)
        model_dir = final if dtype == "float32" else cast_model(final, tmp / "model", dtype)
        operations = [CommitOperationAdd(path.name, str(path)) for path in sorted(model_dir.iterdir())
                      if path.is_file()]
        operations += [CommitOperationAdd(f"training/{name}", str(run_dir / name)) for name in TRAINING_RECORDS
                       if (run_dir / name).exists()]
        card = tmp / "README.md"
        card.write_text(model_card(run_dir, repo_id, dtype=dtype, base_model=base_model,
                                   base_revision=base_revision), encoding="utf-8")
        operations.append(CommitOperationAdd("README.md", str(card)))
        info = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                 commit_message=commit_message or f"CBG v2 run {run_dir.parent.name} ({dtype})")
    record = dict(repo_id=repo_id, url=str(repo_url), private=private, dtype=dtype,
                  commit=getattr(info, "oid", None), commit_url=getattr(info, "commit_url", None),
                  files=[op.path_in_repo for op in operations])
    write_json(run_dir / "hf_upload.json", record)
    return record
