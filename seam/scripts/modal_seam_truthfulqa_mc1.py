"""Detached, zero-shot TruthfulQA MC1 evaluation of the unmodified SEAM base model."""

import json
import os
import re
from importlib.metadata import version
from pathlib import Path

import modal

repo = Path(__file__).resolve().parent.parent
app = modal.App("seam-base-truthfulqa-mc1")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(repo / "requirements.txt"))
    .pip_install("wandb==0.19.8", "requests==2.32.3")
)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("seam-mc1-results", create_if_missing=True)
MODEL = "meta-llama/Llama-2-7b-chat-hf"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/hf": hf_volume, "/results": results_volume},
)
def evaluate_base_mc1(run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, _ or -")
    output = Path("/results") / f"base-truthfulqa-mc1-{run_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.environ.update({"HF_HOME": "/hf", "WANDB_MODE": "disabled",
                       "TOKENIZERS_PARALLELISM": "false"})
    status = {"run_id": run_id, "state": "running", "model": MODEL,
              "task": "truthfulqa_mc1", "attack": False}
    write_json(output / "status.json", status)
    results_volume.commit()
    try:
        import torch
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        model.to("cuda:0")
        wrapped = HFLM(pretrained=model, tokenizer=tokenizer, trust_remote_code=True)
        report = lm_eval.simple_evaluate(
            model=wrapped, tasks=["truthfulqa_mc1"], batch_size=8,
            num_fewshot=0,
        )
        task = report["results"]["truthfulqa_mc1"]
        accuracy = float(task["acc,none"])
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"Invalid MC1 accuracy: {accuracy}")
        metrics = {
            "model_name": MODEL, "attack": False, "task": "truthfulqa_mc1",
            "num_fewshot": 0, "batch_size": 8, "accuracy": accuracy,
            "accuracy_percent": 100 * accuracy,
            "accuracy_stderr": task.get("acc_stderr,none"),
            "lm_eval_version": version("lm_eval"),
            "transformers_version": version("transformers"),
            "task_result": task,
            "task_config": report.get("configs", {}).get("truthfulqa_mc1"),
        }
        write_json(output / "metrics.json", metrics)
        status.update({"state": "complete", "accuracy_percent": 100 * accuracy})
        write_json(output / "status.json", status)
        results_volume.commit()
        hf_volume.commit()
        return {"result": str(output), "accuracy_percent": 100 * accuracy}
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def main(run_id: str = ""):
    if not run_id:
        raise ValueError("Pass a unique --run-id")
    call = evaluate_base_mc1.spawn(run_id)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: seam-mc1-results/base-truthfulqa-mc1-{run_id}")
