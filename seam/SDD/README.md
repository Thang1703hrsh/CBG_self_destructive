# SDD: Self-Degraded Defense against Malicious Fine-tuning

Official code for the paper **“SDD: Self-Degraded Defense against Malicious Fine-tuning.”**

In this repository, we release:
- our **training framework**
- our **evaluation framework**
- our **dataset construction pipeline**

## Repository Layout

- `data/`  
  Place raw community datasets here (see **Data**), then run `toxity_dataset_construction.py` to build the processed dataset used in our experiments.

- `eval/`  
  Code for **GPT-4-based evaluation** (judge evaluation).

  For **general capability evaluation**, we use the `lm-evaluation-harness` (LM Eval).  
  For **LLM-Finetune-Safety** evaluation, please follow the official implementation linked below.

- `train/`  
  Training code built on top of **LLaMA-Factory** (v0.8.0).  
  Hyperparameter choices and training setups are described in the paper.

- `toxity_dataset_construction.py`  
  Implements the dataset construction procedure used in SDD.

## Data

We use several **community datasets** (e.g., AdvBench, HEx-PHI, BeaverTails).  
These datasets may require accepting their license/agreement and/or obtaining authorization from the dataset authors/maintainers.  
**Therefore, we do NOT redistribute the original dataset files in this repo.**

Please download them from their official sources and place them under the `data/` folder, then run:

```bash
python toxity_dataset_construction.py
