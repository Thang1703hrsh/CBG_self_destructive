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
# Host (container) RAM for the GPU stage, in MiB. A Modal container OOM (exit 137,
# "killed ... memory") is a host-RAM limit, not a CUDA OOM; raise this for it.
GPU_STAGE_MEMORY = int(os.environ.get("CBG_MODAL_MEMORY", "65536"))

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
# The author-supplied reference is optional at image-build time, but mandatory
# for a complete S0.1 pass. Never replace it with a fabricated reference file.
for reference_name in ("cbg_reference.py", "verify_cbg_v2.py"):
    if (ROOT / reference_name).is_file():
        image = image.add_local_file(str(ROOT / reference_name), REMOTE + "/" + reference_name)
app = modal.App("cbg-spec-s0")
outputs = modal.Volume.from_name("cbg-spec-s0-output", create_if_missing=True)
data = modal.Volume.from_name("cbg-spec-data", create_if_missing=True)
cache = modal.Volume.from_name("cbg-spec-hf", create_if_missing=True)
volumes = {OUTPUT: outputs, DATA: data, CACHE: cache}
hf_secret = modal.Secret.from_name("HF_TOKEN")

REP_NOISE_REVISION = "11ea63756195527164a9aa2850f7544bd251eab1"
REP_NOISE_SHA256 = "c3c92961d834afe3ecbfbfa29e87ca759b90a73498aaf807162328e0ebe8e4ee"
ALPACA_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"
BEAVERTAILS_REVISION = "325c10601deaf84618333d9624a0637543e9ae62"


def validate_run_name(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run_name must be 1--80 letters/digits/underscores/hyphens")


def s02_calibration_config(config):
    """The radius geometry is independent of alpha and beta."""
    return {key: value for key, value in config.items()
            if key not in {"alpha", "beta", "r", "perturb_precision", "log_gradient_components"}}


def s02_calibration_key(config):
    from src.experiments.protocol import digest
    return digest(s02_calibration_config(config))[:12]


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


def run_command(argv, log_path, extra_env=None):
    """Stream logs and close file handles before committing the volume."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"argv": argv}), flush=True)
    env = dict(os.environ, **(extra_env or {}))
    with Path(log_path).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=REMOTE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
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
def preflight(protocol, choices, run_name, stage, skip_s02_gate=False):
    os.chdir(REMOTE)
    from src.experiments.cli import code_fingerprint, qualification_fingerprint
    from src.experiments.protocol import file_digest, read_json, validate_manifest
    from src.experiments.scenarios import defense_config, load_protocol
    require_verification(run_name)
    p = remote_protocol(protocol, run_name)
    base = Path(OUTPUT) / run_name
    if stage in {"s02", "s03"}:
        audit = read_json(base / "s00" / "length_audit.json")
        safety = audit["safety"]
        benign = audit["benign"]
        if (safety["limit"] != p["defense"]["safety_max_length"]
                or benign["limit"] != p["defense"]["benign_max_length"]):
            raise ValueError(
                "S0.0 audit used different sequence limits from the current protocol"
            )
        if audit.get("status") != "passed":
            print("WARNING: S0.0 truncation gate overridden for this pilot: "
                  f"safety {safety['affected_fraction']:.2%}, "
                  f"benign {benign['affected_fraction']:.2%} affected. "
                  "S0.2/S0.3 describe the truncated 256/128-token protocol only.", flush=True)
    prepared = load_protocol(base / "prepared_protocol.yaml")
    if qualification_fingerprint(prepared) != qualification_fingerprint(p):
        raise ValueError("pilot structure/data/acceptance criteria changed; prepare a new run")
    validate_manifest(read_json(p["manifest"]))
    resolved = defense_config(p, choices, model=p["models"]["pilot"], seed=p["seeds"]["defense"][0],
                              output_dir=str(base / "unused"), pilot=True)
    from dataclasses import asdict
    from src.core.cbg_trainer import CbgArguments
    expected = asdict(CbgArguments(**resolved["CbgArguments"]).spec_config())
    prepared_config = read_json(base / "prepared_method.json")
    if stage not in {"s02", "s03"} and dict(prepared_config, r=expected["r"]) != expected:
        raise ValueError("v2 method settings changed since prepare; use a new run name")
    if stage == "s03":
        if skip_s02_gate:
            s01 = read_json(base / "s01" / "summary.json")
            pinned = read_json(base / "model_revision.json")
            if (s01.get("status") != "passed" or s01.get("model") != pinned["model"]
                    or s01.get("model_revision") != pinned["revision"]
                    or pinned["model"] != p["models"]["pilot"]
                    or not (base / "bank.pt").is_file()):
                raise ValueError("S0.1 pass and the prepared model/bank are still required for exploratory S0.3")
            print("WARNING: S0.2 gate explicitly skipped; this S0.3 run is exploratory, "
                  "not radius-qualified, and cannot qualify later stages.", flush=True)
        else:
            precision = choices.get("spec_perturb_precision", "fp32")
            report = read_json(base / "s02" / f"{s02_calibration_key(expected)}-{precision}.json")
            if (report.get("status") != "passed" or len(report.get("draws", [])) != 16
                    or report.get("exact_precision") != "fp32"
                    or report.get("code_sha256") != code_fingerprint()
                    or report.get("model_revision") != read_json(base / "model_revision.json")["revision"]
                    or report.get("manifest_sha256") != read_json(p["manifest"])["sha256"]
                    or report.get("bank_sha256") != file_digest(base / "bank.pt")
                    or s02_calibration_config(report["config"]) != s02_calibration_config(expected)):
                raise ValueError("matching, current 16-draw S0.2 calibration required before S0.3")
            selected = choices["relative_radius"]
            if not any(row["radius"] == selected and row["passes_threshold"] for row in report["summary"]):
                raise ValueError("selected S0.3 radius did not meet the S0.2 thresholds")
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
        if (dict(radius["config"], r=selected) != expected
            or radius.get("perturb_precision") != expected["perturb_precision"]):
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


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={OUTPUT: outputs})
def s01_reference(run_name):
    """Run the author's reference separately; missing is not a passing result."""
    validate_run_name(run_name)
    os.chdir(REMOTE)
    from src.experiments.protocol import file_digest, write_json
    base = Path(OUTPUT) / run_name / "s01"
    report = {"status": "missing_reference", "required": "cbg_reference.py", "scripts": {}}
    try:
        for name in ("cbg_reference.py", "verify_cbg_v2.py"):
            source = Path(REMOTE) / name
            if not source.is_file():
                continue
            report["scripts"][name] = {"sha256": file_digest(source), "status": "running"}
            run_command([sys.executable, str(source)], base / (name + ".log"))
            report["scripts"][name]["status"] = "passed"
        if report["scripts"].get("cbg_reference.py", {}).get("status") == "passed":
            report["status"] = "passed"
        return report
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        write_json(base / "reference.json", report)
        outputs.commit()


@app.function(image=image, gpu="H100", cpu=4, memory=131072, timeout=2*3600,
              volumes={OUTPUT: outputs, CACHE: cache}, secrets=[hf_secret])
def s01_h100(options, run_name):
    """L1--L6 on one H100; no optimizer or dataset preparation."""
    validate_run_name(run_name)
    os.chdir(REMOTE)
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM
    from src.experiments.cli import code_fingerprint, write_yaml_once
    from src.experiments.protocol import read_json, write_json
    from src.experiments.s01 import run_model_checks
    require_verification(run_name)
    if options.get("scenario") != "S0.1" or options.get("method_revision") != "cbg_v2":
        raise ValueError("an S0.1 CBG v2 configuration is required")
    if options.get("gpu") != "H100" or torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("S0.1 must run on exactly one H100")
    if options.get("model") != "meta-llama/Llama-3.2-1B-Instruct":
        raise ValueError("this S0.1 setup is pinned to the guide's Llama-3.2-1B-Instruct pilot")
    base = Path(OUTPUT) / run_name / "s01"
    if (base / "model_checks.json").exists():
        raise ValueError("S0.1 already has results under this run name; choose a new run name")
    write_yaml_once(base / "config.yaml", options)
    summary = dict(status="running", scenario="S0.1", code_sha256=code_fingerprint(),
                   output=str(base), model=options["model"], reference=read_json(base / "reference.json"))
    try:
        revision = HfApi().model_info(options["model"]).sha
        summary["model_revision"] = revision
        write_json(base / "summary.json", summary)
        model = AutoModelForCausalLM.from_pretrained(options["model"], revision=revision,
            torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0")
        torch.manual_seed(options["seed"])
        result = run_model_checks(model, options, base / "model_checks.json")
        summary["model_checks_status"] = result["status"]
        summary["gpu"] = result["gpu"]
        summary["status"] = ("passed" if summary["reference"]["status"] == "passed"
                             else "incomplete_missing_reference")
        return summary
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(base / "summary.json", summary)
        outputs.commit()
        cache.commit()


@app.function(image=image, cpu=2, memory=16384, timeout=4*3600,
              volumes=volumes, secrets=[hf_secret])
def s00_audit(protocol, run_name):
    """Audit full prompt+response token lengths on the frozen defense pools."""
    validate_run_name(run_name)
    os.chdir(REMOTE)
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from src.experiments.cli import code_fingerprint
    from src.experiments.length_audit import audit_lengths
    from src.experiments.protocol import digest, prepare_splits, read_json, records, write_once

    p = remote_protocol(protocol, run_name)
    if p["models"]["pilot"] != "meta-llama/Llama-3.2-1B-Instruct":
        raise ValueError("S0.0 v2 must audit the Llama-3.2-1B-Instruct pilot")
    defense = p["defense"]
    for key in ("safety_max_length", "benign_max_length"):
        value = defense[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{key} must be an integer >= 2")

    base = Path(OUTPUT) / run_name
    try:
        summary_path = base / "s01" / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if summary.get("model") != p["models"]["pilot"] or not summary.get("model_revision"):
                raise ValueError("S0.1 summary belongs to a different model or has no pinned revision")
            revision = summary["model_revision"]
        else:
            revision = HfApi().model_info(p["models"]["pilot"]).sha

        dataset = p["data"]
        manifest = prepare_splits(dataset["safety"], dataset["benign"], dataset["evaluation"],
            seed=p["seeds"]["split"], refusal_model=dataset["refusal_model"],
            **{key: dataset[key] for key in ("defense_size", "benign_size", "validation_size",
                                               "attack_size", "max_rows_per_prompt")})
        tokenizer = AutoTokenizer.from_pretrained(p["models"]["pilot"], revision=revision)
        audit = audit_lengths(records(dataset["safety"]), records(dataset["benign"]),
            manifest["defense_indices"], manifest["benign_indices"], tokenizer,
            safety_max_length=defense["safety_max_length"],
            benign_max_length=defense["benign_max_length"])
        result = dict(scenario="S0.0", model=p["models"]["pilot"], model_revision=revision,
                      split_seed=p["seeds"]["split"], manifest_sha256=manifest["sha256"],
                      source_sha256={key: item["sha256"] for key, item in manifest["sources"].items()},
                      protocol_sha256=digest(p), code_sha256=code_fingerprint(), **audit)
        write_once(base / "s00" / "length_audit.json", result)
        return result
    finally:
        outputs.commit()
        cache.commit()


@app.function(image=image, gpu="H100", cpu=4, memory=131072, timeout=24*3600,
              volumes=volumes, secrets=[hf_secret])
def s02_radius(protocol, choices, run_name, precision):
    """S0.2: fixed training batch, 16 draws, FP32 exact HVP, one compute mode."""
    if precision not in {"bf16", "tf32", "fp32"}:
        raise ValueError("S0.2 precision must be bf16, tf32 or fp32")
    validate_run_name(run_name)
    os.chdir(REMOTE)
    import math
    import statistics
    import torch
    from dataclasses import asdict, replace
    from huggingface_hub import snapshot_download
    from src.cbg.spec import _forward, joint_batch, measurements, spec_precision
    from src.core.cbg_trainer import CbgArguments
    from src.experiments.cli import code_fingerprint
    from src.experiments.geometry import radius_curve, squared_norm
    from src.experiments.protocol import file_digest, read_json, validate_manifest, write_json
    from src.experiments.runner import load_model, move_batch
    from src.experiments.scenarios import defense_config

    require_verification(run_name)
    p = remote_protocol(protocol, run_name)
    base = Path(OUTPUT) / run_name
    s01_report = read_json(base / "s01" / "summary.json")
    if s01_report.get("status") != "passed":
        raise ValueError("S0.1 must pass before S0.2")
    s00_report = read_json(base / "s00" / "length_audit.json")
    manifest = read_json(p["manifest"])
    validate_manifest(manifest)
    bank_path = base / "bank.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if bank["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("geometry bank does not match the frozen data split")
    pinned = read_json(base / "model_revision.json")
    if pinned["model"] != p["models"]["pilot"]:
        raise ValueError("prepared model differs from S0.2 pilot")
    if (s01_report.get("model_revision") != pinned["revision"]
            or s00_report.get("model_revision") != pinned["revision"]
            or s00_report.get("manifest_sha256") != manifest["sha256"]):
        raise ValueError("S0.0/S0.1 model or data revision differs from the prepared pilot")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("S0.2 must run on one H100")
    snapshot = snapshot_download(pinned["model"], revision=pinned["revision"], local_files_only=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model, tokenizer = load_model(snapshot, "cuda:0")
    model.requires_grad_(True)
    from src.experiments.protocol import digest
    if bank["tokenizer_sha256"] != digest(tokenizer.get_vocab()):
        raise ValueError("geometry bank tokenizer mismatch")
    config = CbgArguments(**defense_config(p, choices, model=pinned["model"],
        seed=p["seeds"]["defense"][0], output_dir="unused", pilot=True)["CbgArguments"]).spec_config()
    config = replace(config, alpha=1.0, beta=0.01, perturb_precision=precision)
    batch = move_batch(bank["batches"][0], "cuda:0")
    joint = joint_batch(batch)
    with spec_precision(model, "fp32"), torch.no_grad():
        generator = torch.Generator(device="cuda:0").manual_seed(bank["seed"])
        logits = _forward(model, joint)
        fixed_samples = measurements(logits, joint, fisher=config.fisher,
            generator=generator, logprob_chunk_size=config.logprob_chunk_size).samples.detach()
        del logits
    radii = (1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    output = base / "s02" / f"{s02_calibration_key(asdict(config))}-{precision}.json"
    receipt = dict(status="running", scenario="S0.2", model=pinned["model"],
                   model_revision=pinned["revision"], code_sha256=code_fingerprint(),
                   bank_sha256=file_digest(bank_path), manifest_sha256=manifest["sha256"],
                   config=asdict(config), radii=radii, precision=precision,
                   exact_precision="fp32", required_draws=16,
                   length_audit=dict(status=s00_report["status"],
                       safety_limit=s00_report["safety"]["limit"],
                       safety_affected_fraction=s00_report["safety"]["affected_fraction"],
                       benign_limit=s00_report["benign"]["limit"],
                       benign_affected_fraction=s00_report["benign"]["affected_fraction"]),
                   draws=[])
    if output.exists():
        old = read_json(output)
        if any(old.get(key) != receipt[key] for key in receipt if key not in {"draws", "status"}):
            raise ValueError("S0.2 receipt settings changed; use a new run name")
        if len(old.get("draws", [])) == 16 and old.get("status") in {"passed", "failed_threshold"}:
            return {key: old[key] for key in ("status", "precision", "summary", "required_draws")}
        print("S0.2 incomplete receipt found; restarting all 16 draws to measure sign noise", flush=True)
    exact_sum = {}
    try:
        for draw in range(16):
            seed = bank["seed"] + draw
            print(f"S0.2 {precision}: draw {draw+1}/16, seed={seed}", flush=True)
            curve = radius_curve(model, batch, config, radii, seed=seed,
                perturb_precision=precision, exact_precision="fp32", microbatches=8,
                output_deltas=(), exact_accumulator=exact_sum,
                fixed_samples=fixed_samples)
            receipt["draws"].append(curve)
            write_json(output, receipt)
            outputs.commit()
            torch.cuda.empty_cache()
        mean_exact_norm_sq = float(squared_norm(exact_sum)) / (16*16)
        mean_draw_norm_sq = statistics.mean(draw["exact_hvp_norm"]**2 for draw in receipt["draws"])
        sign_std = math.sqrt(max(0.0, mean_draw_norm_sq - mean_exact_norm_sq))
        summary = []
        for index, radius in enumerate(radii):
            rows = [draw["curve"][index] for draw in receipt["draws"]]
            if any(row.get("status") != "ok" for row in rows):
                summary.append(dict(radius=radius, passes_threshold=False,
                                    failed_draws=sum(row.get("status") != "ok" for row in rows)))
                continue
            errors = [row["relative_error"] for row in rows]
            cosines = [row["cosine"] for row in rows]
            mean_error = statistics.mean(errors)
            mean_cosine = statistics.mean(cosines)
            mean_absolute_error = statistics.mean(row["absolute_error_norm"] for row in rows)
            max_linearity_error = max(row["linearity_error"] for row in rows)
            summary.append(dict(radius=radius, mean_relative_error=mean_error,
                                mean_cosine=mean_cosine,
                                mean_absolute_error_norm=mean_absolute_error,
                                sign_draw_std_norm=sign_std,
                                error_to_sign_std=(mean_absolute_error/sign_std if sign_std > 0 else None),
                                error_std=statistics.stdev(errors),
                                cosine_std=statistics.stdev(cosines),
                                max_linearity_error=max_linearity_error,
                                passes_threshold=mean_error <= 0.2 and mean_cosine >= 0.9
                                and sign_std > 0 and mean_absolute_error <= 0.25*sign_std
                                and max_linearity_error <= 0.01
                                and all(math.isfinite(value) for value in errors + cosines)))
        receipt["summary"] = summary
        receipt["status"] = "passed" if any(row["passes_threshold"] for row in summary) else "failed_threshold"
        return {key: receipt[key] for key in ("status", "precision", "summary", "required_draws")}
    finally:
        write_json(output, receipt)
        outputs.commit()
        cache.commit()


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
        resolved = defense_config(p, choices, model=p["models"]["pilot"], seed=p["seeds"]["defense"][0],
                                  output_dir=str(base / "unused"), pilot=True)
        from dataclasses import asdict
        from src.core.cbg_trainer import CbgArguments
        d = p["data"]
        manifest = prepare_splits(d["safety"], d["benign"], d["evaluation"],
            seed=p["seeds"]["split"], refusal_model=d["refusal_model"],
            **{k:d[k] for k in ("defense_size", "benign_size", "validation_size", "attack_size",
                                "max_rows_per_prompt")})
        # Freeze the protocol only after every input file and split validates.
        # A failed data check must not leave a partial run identity behind.
        write_yaml_once(base / "prepared_protocol.yaml", p)
        write_once(base / "prepared_method.json", asdict(CbgArguments(**resolved["CbgArguments"]).spec_config()))
        write_once(p["manifest"], manifest)
        receipt = base / "model_revision.json"
        from src.experiments.protocol import read_json
        if receipt.exists():
            pinned = read_json(receipt)
            if pinned["model"] != p["models"]["pilot"]:
                raise ValueError("run name already belongs to another pilot model")
            revision = pinned["revision"]
        else:
            s01_summary = base / "s01" / "summary.json"
            if s01_summary.exists() and read_json(s01_summary).get("status") == "passed":
                summary = read_json(s01_summary)
                if summary.get("model") != p["models"]["pilot"]:
                    raise ValueError("S0.1 pilot model differs from the prepared model")
                revision = summary["model_revision"]
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


@app.function(image=image, gpu=GPU, cpu=4, memory=GPU_STAGE_MEMORY, timeout=24*3600,
              volumes=volumes, secrets=[hf_secret])
def gpu_stage(stage, protocol, choices, run_name, resume, mem_snapshot=False, skip_s02_gate=False):
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
    if stage == "s03":
        from src.experiments.protocol import digest
        s03_folder = "s03_exploratory" if skip_s02_gate else "s03"
        receipt_base = base / s03_folder / digest(choices)[:12]
        audit = read_json(base / "s00" / "length_audit.json")
        write_json(receipt_base / "length_audit.json", audit)
        write_json(receipt_base / "qualification.json", dict(
            status="unqualified_s02_skipped" if skip_s02_gate else "s02_passed",
            scenario="S0.3", s02_gate_skipped=skip_s02_gate,
            selected_radius=choices["relative_radius"],
            selected_precision=choices.get("spec_perturb_precision", "fp32"),
            model=pinned["model"], model_revision=pinned["revision"],
            manifest_sha256=read_json(p["manifest"])["sha256"]))
    else:
        receipt_base = base
    write_yaml_once(receipt_base / f"protocol-{stage}.yaml", p)
    write_yaml_once(receipt_base / f"choices-{stage}.yaml", choices)
    write_json(receipt_base / f"hardware-{stage}.json", hardware())
    try:
        if stage == "radius":
            result = run_probe(p, snapshot, str(base / f"choices-{stage}.yaml"), str(base / "bank.pt"),
                               str(base / "radius.json"), device="cuda:0", radius=True,
                               perturb_precision=choices.get("spec_perturb_precision", "fp32"))
            result.update(model=pinned["model"], model_revision=pinned["revision"])
            write_json(base / "radius.json", result)
            return result
        if stage not in {"probe", "train", "s03"}:
            raise ValueError("GPU stage must be radius, probe, train or s03")
        if stage == "s03":
            out = receipt_base
        else:
            out = base / ("probe" if stage == "probe" else "pilot")
        if (out / "train_log.jsonl").exists() and not resume:
            raise ValueError("this stage already has training logs; use --resume or a new run-name")
        cfg = defense_config(p, choices, model=pinned["model"], seed=p["seeds"]["defense"][0],
                             output_dir=str(out), pilot=True)
        cfg["CbgArguments"]["model_revision"] = pinned["revision"]
        if stage == "probe":
            # One smoke-test step. Do not write a checkpoint: with save_steps=1 the
            # trainer serializes fp32 weights + Adam state after step 1, which costs
            # host RAM (several times the model size), disk and time, and is never
            # used by a probe.
            cfg["TrainingArguments"].update(max_steps=1, save_strategy="no")
        else:
            cfg["TrainingArguments"].update(save_steps=min(25, cfg["TrainingArguments"]["max_steps"]),
                                            save_total_limit=2)
        write_yaml_once(out / "config.yaml", cfg)
        command = [sys.executable, "-m", "src.train", "--method", "cbg", "--config", str(out / "config.yaml")]
        if resume:
            if not re.fullmatch(r"checkpoint-\d+", resume):
                raise ValueError("resume must name checkpoint-N inside this run")
            command += ["--cbg_resume_from_checkpoint", str(out / resume)]
        extra_env = {}
        if mem_snapshot:
            # Read by the allocator-history hook in src/train.py (see reply).
            extra_env["CBG_MEM_SNAPSHOT_PATH"] = str(out / "cuda_memory_snapshot.pickle")
        run_command(command, out / "console.log", extra_env)
        return dict(status="complete", stage=stage, output=str(out))
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(stage: str = "verify", run_name: str = "s0-first",
         protocol: str = "config/cbg_scenarios.yaml", choices: str = "pilot_choices.yaml", resume: str = "",
         mem_snapshot: bool = False, s01_config: str = "config/cbg_method_v2_s01_h100.yaml",
         s02_precision: str = "bf16", beta: float = 0.0, alpha: float = -1.0,
         kappa_mu: float = 0.0, selected_radius: float = 0.0,
         selected_precision: str = "", skip_s02_gate: bool = False):
    validate_run_name(run_name)
    if skip_s02_gate and stage != "s03":
        raise ValueError("--skip-s02-gate is only valid for exploratory S0.3")
    if skip_s02_gate and (selected_radius <= 0 or not selected_precision):
        raise ValueError("exploratory S0.3 requires explicit --selected-radius and --selected-precision")
    if stage == "verify":
        result = verify.remote(run_name)
    elif stage == "s00":
        import yaml
        p = yaml.safe_load(Path(protocol).read_text(encoding="utf-8"))
        prepare_data.remote()  # CPU only; reuses verified pinned data when cached.
        result = s00_audit.remote(p, run_name)
    elif stage == "s01":
        import yaml
        options = yaml.safe_load(Path(s01_config).read_text(encoding="utf-8"))
        verify.remote(run_name)  # CPU checks pass before allocating the H100.
        reference = s01_reference.remote(run_name)
        if reference["status"] != "passed":
            print("cbg_reference.py is missing: H100 L1-L6 will run, but the full S0.1 gate remains incomplete.", flush=True)
        result = s01_h100.remote(options, run_name)
    elif stage == "data":
        result = prepare_data.remote()
    else:
        import yaml
        p = yaml.safe_load(Path(protocol).read_text(encoding="utf-8"))
        from src.experiments.cli import load_choices
        c = load_choices(choices)
        if stage in {"s02", "s03"}:
            if beta < 0 or alpha < -1 or kappa_mu < 0 or selected_radius < 0:
                raise ValueError("beta, kappa_mu, radius must be nonnegative; alpha must be >= -1")
            if beta:
                c["beta"] = beta
            if alpha >= 0:
                c["alpha"] = alpha
            if kappa_mu:
                c["kappa_mu"] = kappa_mu
            if selected_radius:
                c["relative_radius"] = selected_radius
            if selected_precision:
                if selected_precision not in {"bf16", "tf32", "fp32"}:
                    raise ValueError("selected_precision must be bf16, tf32 or fp32")
                c["spec_perturb_precision"] = selected_precision
            if stage == "s03":
                c["log_gradient_components"] = True
        if not {"alpha", "beta", "xi", "kappa_mu", "relative_radius"} <= set(c):
            raise ValueError("S0 choices require alpha, beta, xi, kappa_mu, relative_radius; v2 options are allowed")
        if stage == "prepare":
            result = prepare.remote(p, c, run_name)
        elif stage == "s02":
            preflight.remote(p, c, run_name, "s02")
            result = s02_radius.remote(p, c, run_name, s02_precision)
        elif stage == "s03":
            preflight.remote(p, c, run_name, "s03", skip_s02_gate)
            result = gpu_stage.remote(stage, p, c, run_name, resume, mem_snapshot, skip_s02_gate)
        elif stage in {"probe", "radius", "train"}:
            preflight.remote(p, c, run_name, stage)
            result = gpu_stage.remote(stage, p, c, run_name, resume, mem_snapshot)
        elif stage == "gate":
            result = gate.remote(p, run_name)
        else:
            raise ValueError("stage must be data, verify, s00, s01, s02, s03, prepare, probe, radius, train or gate")
    print(json.dumps(result, indent=2, default=str))
