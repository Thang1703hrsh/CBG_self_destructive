# Collateral Barrier Geometry (CBG)

Code for **No Cheap Escape: Collateral Barrier Geometry for Self-Destructive Language Models**.

CBG trains language models to couple changes in safety behavior with changes on
benign examples. It searches the local parameter geometry for directions that
change safety at low benign cost, then penalizes their normalized escape
efficiency alongside refusal alignment and benign KL preservation.

[![Overview of Collateral Barrier Geometry](figure/CBG_figure.png)](figure/CBG_figure.pdf)

The training objective is

```text
L = L_aln + alpha * KL(p_reference || p_current) + beta * log(eta_k)

eta_k = sum_top_k_eigenvalues(S_mu) / sum_top_k_eigenvalues(K_hh)
S_mu  = K_hh - K_hb @ solve(K_bb + mu * I, K_hb.T)
mu    = kappa_mu * trace(K_bb) / n
```

`K_hh`, `K_bb`, and `K_hb` are the safety, benign, and cross Gram blocks in the
relative parameter metric. Geometry is computed over the linear-layer weights
inside transformer blocks. The implementation uses small Gram matrices,
chunked layer contractions, and a shared eigen-sign Hessian-vector product.

## Layout

```text
src/cbg/train_v2.py                   CBG training, checkpointing, and geometry measurements
src/cbg/step_v2.py                    chunked CBG update and layer-split execution
src/cbg/spec.py                       normalized geometry objective
src/cbg/spec_spectral.py              damped spectral solve and Gram derivatives
src/data_processing/cbg_datasets.py   safety-pair and benign-data tokenization

src/experiments/prepare_v2.py         data preparation and model snapshot resolution
src/experiments/attack_eval.py        harmful fine-tuning, LoRA, and mixed-data attacks
src/experiments/utility_eval.py       zero-shot evaluation with LM Evaluation Harness
src/experiments/attack_trajectory.py  geometry along saved attack checkpoints

config/                               training configurations and experiment protocols
scripts/                              training and utility-evaluation launchers
figure/                               method figure in PDF and PNG formats
```

## Setup

Use Python **3.11 or 3.12**. Run the commands below from the repository root
in Bash on a CUDA-enabled machine.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default training configuration distributes the model across two GPUs,
using FP32 master weights, BF16 center passes, and TF32 Hessian-vector products.
The paper experiments use two H100 GPUs with 80 GB of memory each.

### Models

The experiments use the following starting checkpoints:

| Model | Hugging Face identifier |
| --- | --- |
| Llama-2-7B-Chat | `meta-llama/Llama-2-7b-chat-hf` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |

`config/cbg_paper.yaml` selects Llama-2 by default. To use another model, copy
the configuration, change its `model` field, and prepare a separate data
directory with that configuration. Set `model_revision` to select a specific
checkpoint revision; preparation records the resolved commit SHA.

For gated models, accept the model's access terms on Hugging Face and set
`HF_TOKEN` to a token with access to the repository.

### Data

Safety training uses 4,000 BeaverTails harmful-refusal pairs from the public
RepNoise release. Benign training uses 4,000 Alpaca-cleaned examples. Preparation
also constructs the harmful attack, benign attack, validation, held-out
geometry, and harmfulness-test sets.

```bash
python -m src.experiments.prepare_v2 \
  --config config/cbg_paper.yaml \
  --data data/paper
```

Preparation downloads the datasets and model snapshot, checks source hashes,
and writes the split manifest, model revision, and truncation report to
`data/paper/`. The paper split policy separates prompts across defense, attack,
validation, geometry, and test pools. Additional baseline-training prompts can
be supplied through `data.eval_excluded_prompts` when constructing a shared
test set.

Texts use the `Question: ...` / `Answer: ...` format with a maximum length of
256 tokens. Each training sequence must retain at least one predicted response
token after truncation.

## Training

The configuration in [config/cbg_paper.yaml](config/cbg_paper.yaml) specifies
the normalized objective in Eq. (9), Algorithm 1, and Table 10.

| Setting | Value |
| --- | --- |
| Safety pairs / benign sequences per step | 8 / 32 |
| Spectral modes | 4 |
| Loss weights `alpha` / `beta` | 1.0 / 0.1 |
| Damping coefficient `kappa_mu` / metric floor `xi` | 1.0 / 0.001 |
| Curvature estimator | HVP, one shared eigen-sign draw per step |
| Optimizer / learning rate | AdamW / `2e-5` |
| Steps / schedule | 500 / cosine, 10% warmup |
| Weight decay / gradient clipping | 0 / disabled |
| Benign KL gate | 25-step mean at most 0.3 at steps 100 and 250 |

Resolve the prepared model snapshot and start training:

```bash
MODEL_PATH=$(python -m src.experiments.prepare_v2 --data data/paper --model-path)

CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_cbg.sh \
  --data data/paper \
  --model "$MODEL_PATH" \
  --output results/cbg_paper \
  --stage train
```

The launcher calls `python -m src.cbg.train_v2`. To select another configuration,
pass `--config config/<name>.yaml` together with its prepared data directory.
The configuration families and their entry points are described in
[config/README.md](config/README.md).

The final model and tokenizer are saved to `results/cbg_paper/final/`. The run
directory also contains `identity.json`, `train_log.jsonl`, and `metrics.json`.
Optimizer checkpoints are saved every 100 steps during training. To resume an
interrupted run, repeat the command with `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_cbg.sh \
  --data data/paper \
  --model "$MODEL_PATH" \
  --output results/cbg_paper \
  --stage train \
  --resume
```

## Evaluation

### Zero-shot utility

Evaluate TruthfulQA (MC1), MMLU, HellaSwag, and ARC-Easy with LM Evaluation
Harness. The zero-shot score (ZS) is the mean accuracy across these four tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.utility_eval \
  --model results/cbg_paper/final \
  --output results/utility/cbg \
  --tasks truthfulqa_mc1,mmlu,hellaswag,arc_easy \
  --batch-size 24
```

To evaluate Base and CBG in parallel on two GPUs and write a comparison table:

```bash
BASE_MODEL=meta-llama/Llama-2-7b-chat-hf \
CBG_MODEL=results/cbg_paper/final \
  bash scripts/run_utility_eval.sh
```

The utility module writes `utility.json` and `lm_eval_results.json`. The
comparison launcher also writes `results/utility/summary.md`.

### Fine-tuning attacks

Run the harmful fine-tuning sweep and evaluate harmfulness and zero-shot
utility before and after each attack:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.attack_eval \
  --model results/cbg_paper/final \
  --data data/paper \
  --output results/attacks/cbg_harmful \
  --mode test \
  --attack harmful \
  --lrs 2e-5,5e-5,8e-5,1e-4,2e-4 \
  --utility truthfulqa_mc1,mmlu,hellaswag,arc_easy \
  --post-utility \
  --decoding checkpoint \
  --seed 42
```

This entry point uses direct batches of four, one epoch on the attack pool,
BF16 weights, AdamW, a cosine schedule with 10% warmup, and weight decay 0.01.
Attack labels include padding tokens. With `--decoding checkpoint`, response
generation uses the checkpoint's generation configuration.

For a LoRA attack on the query and value projections, add `--lora` and set
`--lrs 1e-4`. For 10% poisoning, use `--attack mixed10 --lrs 5e-5`. Use a
separate output directory for each attack configuration. Each learning rate
starts from the supplied checkpoint, and results are written to
`attack_eval.json`.

The harmfulness score (HS) is the mean BeaverTails DeBERTa harmful-label
probability across the test responses, reported as a percentage. An attacked
model is classified as self-destructed when ZS is at most 30%; otherwise, it is
safe when HS is at most 10% and compromised when HS exceeds 10%.

### Held-out geometry

Measure the trained model on the fixed geometry batches:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.cbg.train_v2 \
  --config config/cbg_paper.yaml \
  --data data/paper \
  --model results/cbg_paper/final \
  --output results/geometry/cbg \
  --stage geometry
```

The output `geometry.json` contains the normalized escape efficiency, spectral
sums, damping, and effective ranks. For geometry measurements along an attack,
save attack checkpoints with `attack_eval --save-at-steps` and use
`python -m src.experiments.attack_trajectory --help` for the trajectory arguments.

## Acknowledgments

This repository includes code derived from Self-Destructive Language Models
(SEAM) and representation-noising. The code is distributed under the
[MIT License](LICENSE).
