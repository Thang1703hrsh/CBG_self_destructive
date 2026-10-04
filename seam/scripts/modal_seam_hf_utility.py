"""Zero-shot utility evaluation of the private SEAM checkpoint on Modal."""

import json
import os
import re
from importlib.metadata import version
from pathlib import Path

import modal


app = modal.App("seam-hf-utility-eval")
repo = Path(__file__).resolve().parent.parent
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    str(repo / "requirements.txt")
)
hf_volume = modal.Volume.from_name("seam-hf-utility-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("seam-hf-utility-results", create_if_missing=True)

MODEL = "ducthang1703/seam-llama2-7b-seed42-checkpoint500"
BATCH_SIZE = 8
TASK_METRICS = {
    "truthfulqa_mc1": "acc,none",
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "arc_easy": "acc_norm,none",
}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/hf": hf_volume, "/results": results_volume},
)
def evaluate(run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain 1-64 letters, digits, _ or -")

    output = Path("/results") / f"seam-hf-utility-{run_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.environ.update({
        "HF_HOME": "/hf",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    })
    status = {"run_id": run_id, "model": MODEL, "state": "running", "task": "load_model"}
    write_json(output / "status.json", status)
    results_volume.commit()

    try:
        import torch
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Modal secret 'huggingface-secret' must contain HF_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, token=token, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cuda:0")
        model.eval()
        wrapped = HFLM(pretrained=model, tokenizer=tokenizer, trust_remote_code=True)
        hf_volume.commit()

        summary = {
            "model_name": MODEL,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "attack": False,
            "num_fewshot": 0,
            "batch_size": BATCH_SIZE,
            "gpu": "H100:1",
            "lm_eval_version": version("lm_eval"),
            "transformers_version": version("transformers"),
            "metrics": {},
        }
        for task, metric_name in TASK_METRICS.items():
            status["task"] = task
            write_json(output / "status.json", status)
            results_volume.commit()
            print("SEAM_HF_EVAL_TASK", task, flush=True)

            report = lm_eval.simple_evaluate(
                model=wrapped, tasks=[task], batch_size=BATCH_SIZE, num_fewshot=0
            )
            task_result = report["results"][task]
            if metric_name not in task_result:
                raise KeyError(f"Missing {metric_name} in {task} result: {list(task_result)}")
            accuracy = float(task_result[metric_name])
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(f"Invalid {task} accuracy: {accuracy}")

            metrics = {
                "model_name": MODEL,
                "task": task,
                "metric": metric_name,
                "num_fewshot": 0,
                "batch_size": BATCH_SIZE,
                "accuracy": accuracy,
                "accuracy_percent": 100 * accuracy,
                "task_result": task_result,
                "task_config": report.get("configs", {}).get(task),
            }
            write_json(output / "metrics" / f"{task}.json", metrics)
            summary["metrics"][task] = metrics
            write_json(output / "summary.json", summary)
            results_volume.commit()
            print("SEAM_HF_EVAL_SCORE", task, metrics["accuracy_percent"], flush=True)

        summary["mean_accuracy_percent"] = sum(
            item["accuracy_percent"] for item in summary["metrics"].values()
        ) / len(TASK_METRICS)
        write_json(output / "summary.json", summary)
        status.update({"state": "complete", "task": None})
        write_json(output / "status.json", status)
        results_volume.commit()
        hf_volume.commit()
        print("SEAM_HF_EVAL_COMPLETE", output, flush=True)
        return {"result": str(output), "mean_accuracy_percent": summary["mean_accuracy_percent"]}
    except Exception as exc:
        status.update({"state": "failed", "error": repr(exc)})
        write_json(output / "status.json", status)
        results_volume.commit()
        raise


@app.local_entrypoint()
def main(run_id: str = ""):
    if not run_id:
        raise ValueError("Pass a unique --run-id")
    call = evaluate.spawn(run_id)
    print(f"Submitted Modal call: {call.object_id}")
    print(f"Results: seam-hf-utility-results/seam-hf-utility-{run_id}")
