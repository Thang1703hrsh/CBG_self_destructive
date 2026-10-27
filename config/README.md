# Configuration guide

Use `cbg_paper.yaml` with `python -m src.cbg.train_v2` for the single-point
normalized objective in Eq. (9), Algorithm 1, and Table 10. It sets top-k to 4,
alpha to 1, beta to 0.1, differentiated trace damping, one shared eigen-sign
HVP draw per step, 500 AdamW steps, and a 25-step benign-KL gate window.

All other configurations are historical experiments retained so tests and old
run records remain interpretable:

| Config family | Meaning |
| --- | --- |
| `cbg_v2_*_2xh200*.yaml` | Large-model development recipes; names do not identify a single objective. Some `cbg_final` files enable trajectory geometry, cached gradients, and per-tensor balancing. |
| `cbg_spec_pilot.yaml`, `cbg_method_v2_*`, `cbg_scenarios.yaml` | Earlier normalized-method pilots, ablations, and scenario orchestration. |
| `cbg_v5_*`, `cbg_smoke_1b.yaml`, `cbg_llama32_3b.yaml`, `cbg_v2_smoke_1b.yaml` | Experimental raw single-mode Rayleigh objectives, including LoRA-coordinate variants. They differ from the paper's full-U log-eta geometry. |
| `train.yaml`, `eval.yaml` | Inherited SEAM baseline settings for legacy entry points. |
| `*.example.yaml` | Example scenario choices and checkpoint registries. |

The two YAML schemas are distinct. Section-based `model/data/method/execution`
configs go to `src.cbg.train_v2`; `TrainingArguments/CbgArguments` configs go to
the experimental `src.train --method cbg` integration. Do not interchange them.

Changing a recipe does not retroactively change the provenance of an old
checkpoint. The new paper config has not been used here to retrain the reported
models. Pin the desired complete model revision before a reproduction run;
otherwise preparation resolves and records the current immutable revision.
