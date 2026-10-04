"""Modal runner for one SEAM main-result seed.

Usage:
  modal run scripts/modal_seam_main.py::main --stage preflight
  modal run scripts/modal_seam_main.py::main --stage data
  modal run --detach scripts/modal_seam_main.py::main --stage smoke
  modal run --detach scripts/modal_seam_main.py::main --stage train
  modal run scripts/modal_seam_main.py::main --stage eval-loss
"""

import os
from pathlib import Path

import modal

app = modal.App("seam-main-result")
repo = Path('/root/seam') if os.environ.get('MODAL_IS_REMOTE') == '1' else Path(__file__).resolve().parent.parent
preflight_image = modal.Image.debian_slim(python_version="3.12").pip_install("requests==2.32.3")
data_volume = modal.Volume.from_name("seam-main-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("seam-main-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("seam-main-results", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(repo / "requirements.txt"))
    .pip_install("wandb==0.19.8")
    .add_local_file(str(repo / "requirements.txt"), remote_path="/root/seam/requirements.txt")
    .add_local_dir(str(repo / "src"), remote_path="/root/seam/src")
    .add_local_dir(str(repo / "config"), remote_path="/root/seam/config")
)


@app.function(
    image=preflight_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=180,
)
def preflight():
    import requests

    names = sorted(k for k in os.environ if "HF" in k or "HUGGINGFACE" in k)
    print("Hugging Face credential variable names:", names)
    token = next(
        (
            os.environ[k]
            for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_API_TOKEN")
            if os.environ.get(k)
        ),
        None,
    )
    model_url = "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/config.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    model = requests.get(model_url, headers=headers, timeout=30)
    print("Llama-2-7B-Chat config HTTP status:", model.status_code)
    dataset_url = (
        "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
        "main/data/beavertails_with_refusals_train.json"
    )
    data = requests.get(dataset_url, timeout=60)
    print("Public refusal dataset HTTP status:", data.status_code)
    print("Public refusal dataset bytes:", data.headers.get("Content-Length", "unknown"))
    if data.ok:
        records = data.json()
        print("Public refusal dataset records:", len(records))
        print("Public refusal dataset columns:", sorted(records[0]) if records else [])
        print("Records with refusal:", sum(bool(r.get("refusal")) for r in records))
    return {
        "hf_credential_present": bool(token),
        "llama_model_access": model.ok,
        "public_refusal_dataset_access": data.ok,
    }


@app.function(image=preflight_image, volumes={"/data": data_volume}, timeout=180)
def prepare_data(seed: int = 42):
    import hashlib
    import json
    import random
    import requests

    dataset_url = (
        "https://raw.githubusercontent.com/domenicrosati/representation-noising/"
        "main/data/beavertails_with_refusals_train.json"
    )
    response = requests.get(dataset_url, timeout=90)
    response.raise_for_status()
    records = [
        r for r in response.json()
        if r.get("is_safe") is False and r.get("refusal") and r.get("prompt")
    ]
    random.Random(seed).shuffle(records)
    defense, attack = records[:4000], records[4000:5000]
    if len(defense) != 4000 or len(attack) != 1000:
        raise RuntimeError("Not enough disjoint defense and attack records")
    for name, subset in (("defense", defense), ("attack", attack)):
        path = Path("/data") / f"{name}-seed{seed}.json"
        path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
        print(name, "records", len(subset), "SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    data_volume.commit()
    return {"seed": seed, "defense": len(defense), "attack": len(attack)}


@app.function(
    image=train_image,
    gpu="H100",
    cpu=8,
    memory=327680,
    timeout=12 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def train_seam(seed: int = 42, max_steps: int = 20, smoke: bool = False):
    import subprocess
    import sys

    run_name = f"smoke-fp32-seed{seed}" if smoke else f"seam-20steps-fp32-seed{seed}"
    output = f"/results/{run_name}"
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}; use Run-SEAMFull.ps1 for a unique run")
    command = [
        sys.executable, "-m", "src.train",
        "--data_path", f"/data/defense-seed{seed}.json",
        "--output_dir", output,
        "--defense_size", "4000",
        "--benign_seed", str(seed),
        "--max_steps", str(max_steps),
        "--per_device_train_batch_size", "8",
        "--save_strategy", "no" if smoke else "steps",
        "--save_steps", str(max_steps),
        "--report_to", "none",
        "--wandb_run_name", run_name,
    ]
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": "/hf",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
    })
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, cwd="/root/seam", env=environment, check=True)
    hf_volume.commit()
    results_volume.commit()
    return {"run_name": run_name, "output_dir": output, "steps": max_steps}


@app.function(
    image=train_image,
    gpu="H100:2",
    cpu=8,
    memory=65536,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": data_volume, "/hf": hf_volume, "/results": results_volume},
)
def evaluate_heldout_losses(seed: int = 42, sample_count: int = 64):
    import gc
    import json
    import statistics

    os.environ["HF_HOME"] = "/hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    attack_records = json.loads(Path(f"/data/attack-seed{seed}.json").read_text(encoding="utf-8"))[:sample_count]
    harmful_texts = [f"Question: {r['prompt']}\nAnswer:\n{r['response']}" for r in attack_records]
    refusal_texts = [r["refusal"] for r in attack_records]
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=seed)
    benign_records = alpaca.select(range(4000, 4000 + sample_count))
    benign_texts = [
        f"Question: {r['instruction']}\n{r['input']}\nAnswer: {r['output']}"
        for r in benign_records
    ]

    def encode(texts):
        encoded = tokenizer(
            texts, return_tensors="pt", padding="max_length", truncation=True, max_length=256
        )
        encoded["labels"] = encoded["input_ids"].clone()
        encoded["labels"][encoded["attention_mask"] == 0] = -100
        return encoded

    harmful = encode(harmful_texts)
    refusal = encode(refusal_texts)
    benign = encode(benign_texts)

    from src.core.numerics import masked_causal_ce as masked_ce

    def standard_ce(logits, labels):
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1)),
            labels[:, 1:].contiguous().reshape(-1).long(),
        )

    output = {"seed": seed, "sample_count": sample_count, "losses": {}}
    checkpoint = f"/results/seam-20steps-fp32-seed{seed}/checkpoint-20"
    for label, path in (("base", model_id), ("seam_20_steps", checkpoint)):
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cuda:0")
        model.eval()
        model.config.use_cache = False
        losses = {"harmful_sft": [], "refusal_sft": [], "benign_sft": []}
        with torch.inference_mode():
            for start in range(0, sample_count, 8):
                stop = min(start + 8, sample_count)
                h = {k: v[start:stop].to("cuda:0") for k, v in harmful.items()}
                r = {k: v[start:stop].to("cuda:0") for k, v in refusal.items()}
                b = {k: v[start:stop].to("cuda:0") for k, v in benign.items()}
                mask = (h["input_ids"] != r["input_ids"]).float()
                h_logits = model(h["input_ids"], attention_mask=h["attention_mask"]).logits
                r_logits = model(r["input_ids"], attention_mask=r["attention_mask"]).logits
                b_logits = model(b["input_ids"], attention_mask=b["attention_mask"]).logits
                losses["harmful_sft"].append(masked_ce(h_logits, h["labels"], mask).item())
                losses["refusal_sft"].append(masked_ce(r_logits, r["labels"], mask).item())
                losses["benign_sft"].append(standard_ce(b_logits, b["labels"]).item())
        output["losses"][label] = {
            name: {"mean": statistics.mean(values), "batch_values": values}
            for name, values in losses.items()
        }
        print(label, {k: round(v["mean"], 4) for k, v in output["losses"][label].items()}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    path = Path(f"/results/seam-20steps-fp32-seed{seed}/heldout_losses.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    hf_volume.commit()
    results_volume.commit()
    return {k: {m: round(v["mean"], 4) for m, v in losses.items()} for k, losses in output["losses"].items()}


@app.local_entrypoint()
def main(stage: str = "preflight", seed: int = 42):
    if stage == "preflight":
        print(preflight.remote())
    elif stage == "data":
        print(prepare_data.remote(seed))
    elif stage == "smoke":
        print(train_seam.remote(seed, 1, True))
    elif stage == "train":
        print(train_seam.remote(seed, 20, False))
    elif stage == "eval-loss":
        print(evaluate_heldout_losses.remote(seed))
    else:
        raise ValueError(f"Unknown stage: {stage}")
