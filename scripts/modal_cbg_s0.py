"""Modal entry point for implementation-spec CBG, scenario S0 only.

Default action is CPU verification. GPU work requires an explicit stage.
Use config/cbg_scenarios.yaml plus an explicit pilot choices YAML.
See CBG_MODAL_S0_RUNBOOK.md. Existing V5 launchers are separate experiments.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace"
OUTPUT = "/vol/output"
DATA = "/vol/data"
CACHE = "/vol/hf"
GPU = os.environ.get("CBG_MODAL_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONPATH": REMOTE, "USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
          "HF_HOME": CACHE, "HF_DATASETS_CACHE": CACHE + "/datasets",
          "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "offline",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(ROOT / "src"), REMOTE + "/src")
    .add_local_dir(str(ROOT / "tests"), REMOTE + "/tests")
    .add_local_dir(str(ROOT / "config"), REMOTE + "/config")
)
app = modal.App("cbg-spec-s0")
outputs = modal.Volume.from_name("cbg-spec-s0-output", create_if_missing=True)
data = modal.Volume.from_name("cbg-spec-data", create_if_missing=True)
cache = modal.Volume.from_name("cbg-spec-hf", create_if_missing=True)
volumes = {OUTPUT: outputs, DATA: data, CACHE: cache}
hf_secret = modal.Secret.from_name("huggingface-secret")

REP_NOISE_REVISION = "11ea63756195527164a9aa2850f7544bd251eab1"
REP_NOISE_SHA256 = "c3c92961d834afe3ecbfbfa29e87ca759b90a73498aaf807162328e0ebe8e4ee"
ALPACA_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"
BEAVERTAILS_REVISION = "325c10601deaf84618333d9624a0637543e9ae62"


def validate_run_name(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run_name must be 1--80 letters/digits/underscores/hyphens")


def remote_protocol(protocol, run_name):
    """Use stable volume paths; never reuse a workstation manifest remotely."""
    validate_run_name(run_name)
    p = copy.deepcopy(protocol)
    base = f"{OUTPUT}/{run_name}"
    p["output_root"] = base + "/runs"
    p["manifest"] = base + "/splits.json"
    for key, filename in (("safety", "beavertails_cbg_train.json"),
                          ("benign", "alpaca_train.json"), ("evaluation", "beavertails_test.json")):
        p["data"][key] = f"{DATA}/{filename}"
    return p


def run_command(argv, log_path):
    """Stream logs and close file handles before committing the volume."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"argv": argv}), flush=True)
    with Path(log_path).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=REMOTE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f"command exited with status {status}; inspect {log_path}")


def require_verification(run_name):
    from src.experiments.cli import code_fingerprint
    from src.experiments.protocol import read_json
    path = Path(OUTPUT) / run_name / "verification.json"
    report = read_json(path)
    if report.get("status") != "passed" or report.get("code_sha256") != code_fingerprint():
        raise ValueError("run --stage verify for the current code before preparing or allocating a GPU")


@app.function(image=image, cpu=2, memory=8192, timeout=2*3600,
              volumes={DATA: data, CACHE: cache})
def prepare_data():
    """Build pinned, prompt-groupable S0 sources directly in the data Volume."""
    os.chdir(REMOTE)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from src.experiments.protocol import file_digest, read_json, write_json

    base = Path(DATA)
    receipt = base / "cbg_s0_data_receipt_v2.json"
    targets = {
        "safety": base / "beavertails_cbg_train.json",
        "benign": base / "alpaca_train.json",
        "evaluation": base / "beavertails_test.json",
    }
    try:
        if receipt.exists():
            report = read_json(receipt)
            for name, target in targets.items():
                if not target.is_file() or file_digest(target) != report["files"][name]["sha256"]:
                    raise ValueError(f"managed S0 data changed or is missing: {target}")
            return dict(status="cached", receipt=str(receipt),
                        files={name: row["count"] for name, row in report["files"].items()})
        base.mkdir(parents=True, exist_ok=True)
        temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in targets.items()}
        previous_receipt = base / "cbg_s0_data_receipt.json"
        previous = read_json(previous_receipt) if previous_receipt.exists() else None

        # Reuse v1's immutable Alpaca/evaluation files when their receipt still
        # matches; a fresh Volume downloads the same pinned sources below.
        reuse_common = bool(previous)
        for name in ("benign", "evaluation"):
            if (not targets[name].is_file() or not previous
                    or file_digest(targets[name]) != previous["files"][name]["sha256"]):
                reuse_common = False
        if any(targets[name].exists() for name in ("benign", "evaluation")) and not reuse_common:
            raise ValueError("existing Alpaca/evaluation data is not covered by a valid v1 receipt")
        # A worker can be interrupted after atomically replacing the merged
        # file but before writing the receipt.  This path belongs exclusively
        # to schema v2, so rebuild it below instead of making the Volume
        # permanently unrecoverable.

        repnoise_path = base / "beavertails_with_refusals_train.json"
        if (not repnoise_path.is_file() or file_digest(repnoise_path) != REP_NOISE_SHA256):
            safety_source = hf_hub_download(
                repo_id="anonymous4486/repnoise_beavertail", repo_type="dataset",
                revision=REP_NOISE_REVISION, filename="beavertails_with_refusals_train.json")
            repnoise_path = Path(safety_source)
        if file_digest(repnoise_path) != REP_NOISE_SHA256:
            raise ValueError("pinned RepNoise safety artifact failed its published SHA-256")

        if not reuse_common:
            benign_source = hf_hub_download(
                repo_id="yahma/alpaca-cleaned", repo_type="dataset", revision=ALPACA_REVISION,
                filename="alpaca_data_cleaned.json")
            shutil.copyfile(benign_source, temporary["benign"])

        # The pinned repository exposes the 30K test data as a Parquet file,
        # while its legacy dataset builder advertises only the aggregate
        # `train`/`test` names (the latter is the 330K test set). Loading the
        # Parquet file explicitly prevents silently selecting the wrong corpus.
        if not reuse_common:
            evaluation_source = hf_hub_download(
                repo_id="PKU-Alignment/BeaverTails", repo_type="dataset",
                revision=BEAVERTAILS_REVISION, filename="default/30k_test/0000.parquet")
            evaluation = load_dataset("parquet", data_files={"test": evaluation_source}, split="test")
            if len(evaluation) != 3021:
                raise ValueError(f"pinned BeaverTails 30k_test count changed: {len(evaluation)} != 3021")
            temporary["evaluation"].write_text(
                json.dumps([dict(row) for row in evaluation], ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8")

        # RepNoise supplies the externally generated refusal target. BeaverTails
        # 330K supplies enough harmful response rows for the declared 4K/1K/10K
        # pools. Joining by the exact normalized prompt never copies test rows;
        # prepare_splits applies that exclusion before freezing indices.
        from src.experiments.protocol import prompt_key
        repnoise = json.loads(repnoise_path.read_text(encoding="utf-8"))
        refusals = {}
        for row in repnoise:
            key, refusal = prompt_key(row.get("prompt", "")), str(row.get("refusal", "")).strip()
            if key and refusal:
                refusals.setdefault(key, refusal)
        train_source = hf_hub_download(
            repo_id="PKU-Alignment/BeaverTails", repo_type="dataset",
            revision=BEAVERTAILS_REVISION, filename="default/330k_train/0000.parquet")
        train = load_dataset("parquet", data_files={"train": train_source}, split="train")
        merged = []
        for row in train:
            if row.get("is_safe") is not False or not str(row.get("response", "")).strip():
                continue
            item = dict(row)
            item["refusal"] = refusals.get(prompt_key(item.get("prompt", "")), "")
            merged.append(item)
        if len(merged) < 15000 or sum(bool(row["refusal"]) for row in merged) < 4000:
            raise ValueError("pinned merged source cannot supply the declared S0 pools")
        temporary["safety"].write_text(
            json.dumps(merged, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

        loaded = {}
        source_paths = {"safety": temporary["safety"],
                        "benign": targets["benign"] if reuse_common else temporary["benign"],
                        "evaluation": targets["evaluation"] if reuse_common else temporary["evaluation"]}
        for name, path in source_paths.items():
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError(f"{name} source is not a nonempty JSON array")
            loaded[name] = rows
        if not all({"prompt", "response", "refusal", "is_safe"} <= set(row) for row in loaded["safety"]):
            raise ValueError("RepNoise safety data has an unexpected schema")
        if len(loaded["benign"]) < 4000 or not all(
                {"instruction", "input", "output"} <= set(row) for row in loaded["benign"]):
            raise ValueError("Alpaca data has an unexpected schema or fewer than 4,000 rows")
        if not all("prompt" in row for row in loaded["evaluation"]):
            raise ValueError("BeaverTails evaluation data has an unexpected schema")

        temporary["safety"].replace(targets["safety"])
        if not reuse_common:
            temporary["benign"].replace(targets["benign"])
            temporary["evaluation"].replace(targets["evaluation"])
        report = dict(schema_version=2, sources={
            "safety": dict(repo="PKU-Alignment/BeaverTails", revision=BEAVERTAILS_REVISION,
                           filename="default/330k_train/0000.parquet",
                           refusal_repo="anonymous4486/repnoise_beavertail",
                           refusal_revision=REP_NOISE_REVISION,
                           join="NFKC+casefold+whitespace prompt"),
            "benign": dict(repo="yahma/alpaca-cleaned", revision=ALPACA_REVISION,
                           filename="alpaca_data_cleaned.json"),
            "evaluation": dict(repo="PKU-Alignment/BeaverTails", revision=BEAVERTAILS_REVISION,
                               split="30k_test", filename="default/30k_test/0000.parquet"),
        }, files={name: dict(path=str(target), count=len(loaded[name]), sha256=file_digest(target))
                  for name, target in targets.items()})
        write_json(receipt, report)
        return dict(status="downloaded", receipt=str(receipt),
                    files={name: row["count"] for name, row in report["files"].items()})
    finally:
        data.commit()
        cache.commit()


@app.function(image=image, cpu=1, memory=4096, timeout=600, volumes=volumes)
def preflight(protocol, choices, run_name, stage):
    os.chdir(REMOTE)
    from src.experiments.cli import qualification_fingerprint
    from src.experiments.protocol import read_json, validate_manifest
    from src.experiments.scenarios import defense_config, load_protocol
    require_verification(run_name)
    p = remote_protocol(protocol, run_name)
    base = Path(OUTPUT) / run_name
    prepared = load_protocol(base / "prepared_protocol.yaml")
    if qualification_fingerprint(prepared) != qualification_fingerprint(p):
        raise ValueError("pilot structure/data/acceptance criteria changed; prepare a new run")
    validate_manifest(read_json(p["manifest"]))
    defense_config(p, choices, model=p["models"]["pilot"], seed=p["seeds"]["defense"][0],
                   output_dir=str(base / "unused"), pilot=True)
    if stage == "train":
        radius = read_json(base / "radius.json")
        threshold = p["pilot"]["max_fd_relative_error"]
        selected = choices["relative_radius"]
        if threshold is None or p["search"]["relative_radius"] != selected:
            raise ValueError("set the acceptance threshold and the selected radius in both protocol and choices")
        points = sorted(radius["curve"], key=lambda r:r["radius"])
        indices = [i for i,row in enumerate(points) if row["radius"] == selected and row["relative_error"] <= threshold]
        if not indices or not any(points[j]["relative_error"] <= threshold for j in (indices[0]-1, indices[0]+1) if 0 <= j < len(points)):
            raise ValueError("choose a measured radius with an adjacent low-error point before the 100-step pilot")
        from src.experiments.cli import code_fingerprint
        if radius["code_sha256"] != code_fingerprint():
            raise ValueError("radius report predates the current code")
        mapping = {"alpha":"alpha", "beta":"beta", "xi":"xi", "kappa_mu":"kappa_mu"}
        if any(choices[key] != radius["config"][field] for key,field in mapping.items()):
            raise ValueError("scientific settings changed since the radius measurement")
    return {"status": "ready", "stage": stage}


@app.function(image=image, cpu=1, memory=4096, timeout=600, volumes=volumes)
def gate(protocol, run_name):
    os.chdir(REMOTE)
    from src.experiments.cli import validate_pilot
    base = Path(OUTPUT) / run_name
    try:
        return validate_pilot(remote_protocol(protocol, run_name), base / "verification.json",
            base / "radius.json", base / "pilot/train_log.jsonl", base / "gate.json")
    finally:
        outputs.commit()


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={OUTPUT: outputs})
def verify(run_name):
    validate_run_name(run_name)
    os.chdir(REMOTE)
    try:
        from src.experiments.cli import verify as verify_suite
        return verify_suite(f"{OUTPUT}/{run_name}/verification.json")
    finally:
        outputs.commit()


@app.function(image=image, cpu=2, memory=16384, timeout=4*3600,
              volumes=volumes, secrets=[hf_secret])
def prepare(protocol, choices, run_name):
    """Validate splits/settings and cache model weights on CPU, before GPU use."""
    os.chdir(REMOTE)
    from huggingface_hub import HfApi, snapshot_download
    from src.experiments.cli import make_bank, write_yaml_once
    from src.experiments.protocol import prepare_splits, write_once
    from src.experiments.scenarios import defense_config
    require_verification(run_name)
    p = remote_protocol(protocol, run_name)
    base = Path(OUTPUT) / run_name
    try:
        # Validate all scientific settings before model downloads.
        defense_config(p, choices, model=p["models"]["pilot"], seed=p["seeds"]["defense"][0],
                       output_dir=str(base / "unused"), pilot=True)
        d = p["data"]
        manifest = prepare_splits(d["safety"], d["benign"], d["evaluation"],
            seed=p["seeds"]["split"], refusal_model=d["refusal_model"],
            **{k:d[k] for k in ("defense_size", "benign_size", "validation_size", "attack_size",
                                "max_rows_per_prompt")})
        # Freeze the protocol only after every input file and split validates.
        # A failed data check must not leave a partial run identity behind.
        write_yaml_once(base / "prepared_protocol.yaml", p)
        write_once(p["manifest"], manifest)
        receipt = base / "model_revision.json"
        from src.experiments.protocol import read_json
        if receipt.exists():
            pinned = read_json(receipt)
            if pinned["model"] != p["models"]["pilot"]:
                raise ValueError("run name already belongs to another pilot model")
            revision = pinned["revision"]
        else:
            revision = HfApi().model_info(p["models"]["pilot"]).sha
        snapshot = snapshot_download(p["models"]["pilot"], revision=revision)
        # The GPU stage loads this precise snapshot; the original model ID is
        # retained in the scientific protocol and radius qualification receipt.
        write_once(receipt, dict(model=p["models"]["pilot"], revision=revision))
        if not (base / "bank.pt").exists():
            make_bank(p, snapshot, choices, str(base / "bank.pt"), pool="defense", device="cpu")
        return dict(status="prepared", model=p["models"]["pilot"], revision=revision,
                    output=str(base), manifest_sha256=manifest["sha256"])
    finally:
        cache.commit()
        outputs.commit()


@app.function(image=image, gpu=GPU, cpu=4, memory=65536, timeout=24*3600,
              volumes=volumes, secrets=[hf_secret])
def gpu_stage(stage, protocol, choices, run_name, resume):
    os.chdir(REMOTE)
    import torch
    from huggingface_hub import snapshot_download
    from src.experiments.cli import run_probe, write_yaml_once
    from src.experiments.protocol import read_json, validate_manifest, write_json
    from src.experiments.scenarios import defense_config
    from src.experiments.artifacts import hardware
    require_verification(run_name)
    p = remote_protocol(protocol, run_name)
    base = Path(OUTPUT) / run_name
    validate_manifest(read_json(p["manifest"]))
    pinned = read_json(base / "model_revision.json")
    if pinned["model"] != p["models"]["pilot"]:
        raise ValueError("prepared model and requested pilot differ")
    if torch.cuda.device_count() != 1:
        raise ValueError("spec pilot requires one model device; multi-GPU sharding is not implemented")
    snapshot = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    write_yaml_once(base / f"protocol-{stage}.yaml", p)
    write_yaml_once(base / f"choices-{stage}.yaml", choices)
    write_json(base / f"hardware-{stage}.json", hardware())
    try:
        if stage == "radius":
            result = run_probe(p, snapshot, str(base / f"choices-{stage}.yaml"), str(base / "bank.pt"),
                               str(base / "radius.json"), device="cuda:0", radius=True)
            result.update(model=pinned["model"], model_revision=pinned["revision"])
            write_json(base / "radius.json", result)
            return result
        if stage not in {"probe", "train"}:
            raise ValueError("GPU stage must be radius, probe or train")
        out = base / ("probe" if stage == "probe" else "pilot")
        if (out / "train_log.jsonl").exists() and not resume:
            raise ValueError("this stage already has training logs; use --resume or a new run-name")
        cfg = defense_config(p, choices, model=pinned["model"], seed=p["seeds"]["defense"][0],
                             output_dir=str(out), pilot=True)
        cfg["CbgArguments"]["model_revision"] = pinned["revision"]
        if stage == "probe":
            cfg["TrainingArguments"].update(max_steps=1, save_steps=1)
        cfg["TrainingArguments"].update(save_steps=min(25, cfg["TrainingArguments"]["max_steps"]), save_total_limit=2)
        write_yaml_once(out / "config.yaml", cfg)
        command = [sys.executable, "-m", "src.train", "--method", "cbg", "--config", str(out / "config.yaml")]
        if resume:
            if not re.fullmatch(r"checkpoint-\d+", resume):
                raise ValueError("resume must name checkpoint-N inside this run")
            command += ["--cbg_resume_from_checkpoint", str(out / resume)]
        run_command(command, out / "console.log")
        return dict(status="complete", stage=stage, output=str(out))
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(stage: str = "verify", run_name: str = "s0-first",
         protocol: str = "config/cbg_scenarios.yaml", choices: str = "pilot_choices.yaml", resume: str = ""):
    validate_run_name(run_name)
    if stage == "verify":
        result = verify.remote(run_name)
    elif stage == "data":
        result = prepare_data.remote()
    else:
        import yaml
        p = yaml.safe_load(Path(protocol).read_text(encoding="utf-8"))
        c = yaml.safe_load(Path(choices).read_text(encoding="utf-8"))
        if not isinstance(c, dict) or set(c) != {"alpha", "beta", "xi", "kappa_mu", "relative_radius"}:
            raise ValueError("S0 choices must contain exactly alpha, beta, xi, kappa_mu, relative_radius")
        if stage == "prepare":
            result = prepare.remote(p, c, run_name)
        elif stage in {"probe", "radius", "train"}:
            preflight.remote(p, c, run_name, stage)
            result = gpu_stage.remote(stage, p, c, run_name, resume)
        elif stage == "gate":
            result = gate.remote(p, run_name)
        else:
            raise ValueError("stage must be data, verify, prepare, probe, radius, train or gate")
    print(json.dumps(result, indent=2, default=str))
