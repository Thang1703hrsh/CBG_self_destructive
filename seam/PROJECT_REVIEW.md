# Review of the SDD, CTRAP, and SEAM reruns

Review date: 2 October 2026. Scope: the current working tree, including uncommitted changes and the added baseline implementations. This review does not change training code, submit jobs, upload checkpoints, or inspect current remote job state.

**Assessment:** the project has a useful common comparison workflow, but the shared attack labels and harmfulness test construction need correction before the resulting numbers support a reliable comparison. SEAM also has an inherited loss-mask bug that affects defense training. Existing checkpoints should retain their original provenance; fixing evaluation cannot retroactively fix defense training.

## Findings requiring attention

### 1. P1: harmful fine-tuning trains on padding/EOS labels

Locations: [dataset labels](src/data_processing/_datasets.py#L108), [attack trainers](src/eval.py#L135).

The harmful dataset copies every input token into `labels`, including the fixed-length padding. `src.eval` passes this dataset to ordinary `Trainer` without a masking collator. Its default padding collator preserves those labels. `attention_mask=0` does not remove a position from the causal language-model loss; the label must be `-100`.

Confirmed with the actual dataset tokenizer function and a small tokenizer fixture: two padding positions produced labels `[2, 2]`, rather than `[-100, -100]`. The local collator was from Transformers 4.57.6; the pinned 4.49.0 collator and Trainer selection were also checked against their upstream source.

Impact: all three methods' attack sweeps include artificial repeated EOS targets. Short examples contribute more padding, changing attack strength and potentially encouraging early termination. The latter is particularly relevant because the harmfulness scorer assigns empty responses zero harmfulness. The same implementation across methods does not eliminate this confound.

Correction: mask labels using `attention_mask == 0`, preserving real EOS tokens. Decide separately whether the attack is full-sequence or response-only SFT, and record that choice. Existing attack results need new attack runs from the original defended checkpoints if this objective changes; utility-only results are unaffected by this particular issue.

### 2. P1: category extraction shifts rows and silently drops test prompts

Locations: [test category extraction](src/data_processing/_datasets.py#L166), [row pairing](src/data_processing/_datasets.py#L186). The same pattern exists in both training tokenizer functions.

`processed_categories` receives an entry only when a row has a positive category. It is then zipped with the original, unfiltered prompts, responses, and safety flags. Rows without a positive category shorten this list, shift the category assignments, and cause `zip` to stop early.

Confirmed example: input rows were `safe0, unsafe1/violence, safe2, unsafe3/privacy`. The function retained only `unsafe1`, assigned it `privacy`, and omitted `unsafe3`.

Impact: shared HS evaluation can use the wrong subset and incorrect categories. The prefiltered harmful defense/attack inputs avoid safe rows, but are still vulnerable to malformed rows without positive categories. `test_size=0` does not repair the earlier omission.

Correction: process each original row and its category together, or keep an aligned placeholder before filtering. Verify selected prompt IDs and counts against an independently constructed expected test set. Reevaluate HS for existing checkpoints after fixing this.

### 3. P1: the default `all` test filter covers only 11 of 14 categories

Locations: [category mapping](src/data_processing/_datasets.py#L9), [test inclusion condition](src/data_processing/_datasets.py#L190).

The union of `BT_CATEGORIES` omits Animal Abuse, Child Abuse, and Controversial Topics/Politics. The test tokenizer checks the chosen first-positive category against that union even when `test_harmful_subset='all'`. A correctly aligned row whose first-positive category is one of those three is excluded.

This is distinct from the row-alignment bug above. The training tokenizer bypasses category inclusion for `all`, so defense coverage and test coverage are inconsistent. BeaverTails' [official dataset description](https://github.com/PKU-Alignment/beavertails#classification-dataset) lists all 14 categories and states that one example may belong to multiple categories.

Correction: make `all` include every valid harmful row; for subset experiments, check intersection with the row's active categories rather than only its first category. Save category coverage with HS results.

### 4. P1: SEAM's shifted logit mask removes valid answer gradients

Location: [masked cross-entropy](src/core/trainer.py#L871).

The loss masks logits using `mask[..., :-1]`, while masking target labels using `mask[..., 1:]`. At the prompt-to-answer boundary, the answer target can be active while its predicting logit is multiplied by zero. That target still contributes a constant cross-entropy value, but contributes no gradient.

Confirmed with the actual method, a four-token example, and mask `[0, 0, 1, 1]`: the gradient norm for the logit predicting the first selected answer token was exactly zero. Using an all-zero mask returned NaN.

Impact: the unlearning, refusal, and harmful-gradient losses used by SEAM differ from the intended target-masked objective. Long paired texts can become identical after 256-token truncation; batches without valid selected targets have no guard. Nonfinite losses/gradients are not explicitly rejected before the optimizer update. The held-out loss helper repeats this masking logic, so agreement with that helper does not independently validate the objective.

Correction: mask shifted labels without multiplying logits, preserve `-100`, and define how samples/batches with no trainable targets are rejected. Add a boundary-gradient check and an empty-target check. Correcting this changes defense training; evaluating an old checkpoint cannot establish that it was trained with the corrected objective. This code predates the current local edits.

### 5. P2: the utility batch-size setting is ineffective

Locations: [shared utility wrapper](src/utils/metrics.py#L197), [HF utility wrapper](scripts/modal_seam_hf_utility.py#L77), [base MC1 wrapper](scripts/modal_seam_truthfulqa_mc1.py#L55).

These wrappers instantiate `HFLM` without `batch_size`, then supply the desired batch size to `simple_evaluate`. In pinned Harness 0.4.7, `HFLM` defaults to batch size 1, and `simple_evaluate` uses an already constructed LM object as supplied. It does not reconstruct that object with the requested batch size.

Impact: the advertised batch size of 8 or 24 does not control model inference batches, adding avoidable GPU runtime. The HF utility summary records a requested batch size rather than the effective wrapper setting.

Correction: pass `batch_size` when constructing `HFLM`. Keep explicit `num_fewshot=0` and save the actual task configuration. See the pinned [HFLM implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py) and [evaluator](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/evaluator.py). No claim is made here that default MMLU is necessarily few-shot: the checked default template does not specify a nonzero shot count.

### 6. P2: the SEAM full runner reserves two H100s throughout evaluation

Locations: [function allocation](scripts/modal_seam_full.py#L48), [evaluation loop](scripts/modal_seam_full.py#L138).

Preparation, defense training, and all 12 evaluation cases execute within the same `H100:2` function. Restricting evaluation subprocesses to `CUDA_VISIBLE_DEVICES='0'` does not change that function's two-GPU allocation. The manifest describes evaluations seeing one GPU, but this is not a one-GPU evaluation reservation.

Correction: use a CPU coordinator with separate two-H100 defense and one-H100 evaluation workers, as the SDD/CTRAP pipelines already separate their stages. This affects allocation and runtime cost, not the mathematical protocol.

### 7. P2: HF-style CLI forms can silently fall back to YAML values

Location: [CLI override selection](src/utils/config.py#L103).

The parser accepts arguments through argparse but applies a value only if the exact string `--<field>` appears in the raw argument list. Confirmed results with YAML defaults `seed=42, attack=True`:

| Argument | Result |
| --- | --- |
| `--seed 7` | `seed=7` |
| `--seed=7` | `seed=42` |
| `--no_attack` | `attack=True` |

The existing launchers use separate flag/value tokens and explicit `--attack false`, so this does not demonstrate that their current invocations are wrong. It affects manual runs, future launchers, and accepted negative boolean flags.

Correction: merge YAML defaults into parser defaults before parsing, or otherwise track explicitly supplied fields including `--key=value` and negative boolean aliases. Reject unknown configuration fields instead of silently discarding them.

### 8. P2: split reproducibility is weaker than the comparison assumes

Locations: [shared split preparation](scripts/modal_compare_common.py#L30), [SEAM split preparation](scripts/modal_seam_full.py#L92), [comparison validation](scripts/Compare-DefenseResults.ps1#L67).

SEAM and CTRAP cut shuffled source rows into disjoint index ranges, but do not enforce distinct normalized prompts between defense and attack. Duplicate prompts can cross the boundary. The existing split test contains only unique synthetic prompts, so it cannot expose that case. Actual prompt overlap in the remote runs was not measured in this review.

The comparison script checks seed, source SHA, and requested test size. It does not verify actual test IDs, attack-file hashes, code hashes, model/classifier revisions, decoding settings, or effective task configs. Different runs can satisfy its checks while using different evaluators. Source/model/dataset downloads generally use mutable `main` revisions. Much of the added baseline code is currently untracked, and manifests do not capture the working-tree contents.

Correction: save and verify defense/attack/test prompt IDs and hashes, normalized overlap reports, source snapshots/revisions, resolved settings, package versions, and code-content hashes. Preserve an original checkpoint's actual split; do not silently substitute a newly generated split.

### 9. P2: HS generation depends on checkpoint defaults and its evidence is discarded

Locations: [generation parameters](src/utils/metrics.py#L105), [generation call](src/utils/metrics.py#L116), [aggregate HS return](src/utils/metrics.py#L213).

The evaluator sets only `max_new_tokens=50`; sampling, temperature, and beams can come from each checkpoint's generation config. Non-attack cases do not construct a Trainer or explicitly call `set_seed` before HS. Consequently `--seed` is not sufficient to guarantee reproducible HS when generation samples.

`evaluate_harmfulness` produces individual scores and generated texts, but `eval_harmfulness` returns only their average. Empty generations are assigned zero, and category aggregation checks `category` while the dataset emits `categories`.

Correction: make generation settings explicit and common, seed sampled generation, validate the classifier's label mapping/revision, and save per-example prompt/response/score plus empty-output rates. Report HS together with utility and response examples. The Markdown evaluation guide already proposes greedy decoding, but the existing launchers do not implement that proposal.

### 10. P2: optional refusal generation misreads `not harmful` as `harmful`

Location: [judge parsing](data/create_refusal_gpt.py#L101).

`if 'harmful' in result.lower()` also matches `not harmful`. Confirmed by executing the actual async checker with a mocked API response: `not harmful` returned `harmful`.

Impact: this optional data generator can unnecessarily fall back to another model or a generic refusal, changing provenance and adding API calls. It is not invoked by the current public-data Modal launchers, so it does not explain their existing refusal data.

Correction: normalize and match the complete verdict, with an explicit policy for an invalid/failed judge response. The generator's output filename also differs from the default filename expected by the training code.

## Numerical issue requiring a GPU diagnostic

SEAM loads parameters in BF16 and applies the finite-difference perturbation directly to those parameters at [trainer.py:644](src/core/trainer.py#L644). Gradient buffers and difference arithmetic also retain low precision. With a global radius of `1e-3` spread across billions of parameters, many per-weight perturbations can round away.

A synthetic CPU probe with one million BF16 weights around scale 0.02 and a normalized random direction changed approximately 1.5% of weights; the realized displacement norm was approximately `1.94e-4` for a requested `1e-3`. This demonstrates the mechanism, not its measured size in the actual Llama checkpoint. Do not label the real SEAM self-destructive term ineffective solely from this example.

Measure requested versus realized parameter displacement and the norm of the self-destructive gradient in the actual GPU run. Compare a small diagnostic to FP32 perturbation/reduction arithmetic before choosing a correction. Gradient norm divisions also need zero/nonfinite checks. Preserve the distinction between reproducing released code and implementing the paper's mathematical objective.

## What the current baselines implement

| Item | Current behavior | Interpretation |
| --- | --- | --- |
| SEAM defense | 4,000 public RepNoise harmful/refusal pairs; 4,000 shuffled Alpaca-cleaned examples; 500 steps; batch 8; LR `2e-5`; alpha 1, beta .01, epsilon .001 | Follows the public-data option; refusal provenance is not verified as the paper's GPT-4o data. |
| SDD defense | 8,000 unique, categorized BeaverTails `330k_train` prompts, excluding shared attack and `30k_test` prompts; irrelevant answers; 500 steps; nominal batch 24; LR `5e-7` | Reconstructed response-only SFT. Official `SDD/train` is absent. Default AlpacaOnly differs from the paper's Alpaca+LIMA pool; 8-bit AdamW and the chosen template/context are implementation choices. |
| CTRAP defense | Original CTRAPTrainer objective through a data adapter; 5,000 safe pairs, 5,000 harmful pairs, 5,000 UltraChat pairs; LoRA rank 32/alpha 4 | The safe microbatch of 5 plus helpful microbatch of 5 follows the original trainer convention. It is not an accidental batch-halving bug. Chat backbone, source selection, and prompt shortening are declared adaptations. |
| CTRAP duration | H100 full launcher: 20 epochs unless overridden; L40S/train-push defaults: 500 steps | These checkpoints have different defense budgets. At 500 steps, safe/helpful streams each see nominally 2,500 examples and the harmful stream 5,000. Equal steps do not equal data exposure or compute. |
| Shared H100 attack | Fresh base/defended model for each LR; 1,000 attack rows; 250 full-model steps at five LRs | LR attacks are independent rather than cumulative. Subject to finding 1. |
| L40S full variant | LoRA attack rather than full-model attack | Not directly comparable to the H100 full-model sweeps. The comparison script correctly rejects this declared variant. |
| Utility metrics | Full sweeps: TruthfulQA MC2 and raw accuracy; HF utility runner: MC1, normalized HellaSwag/ARC accuracy | Two different summary conventions. The evaluation guide explicitly documents this; the implementations remain different. Do not combine their means as one ZS series. |
| Harmfulness metrics | Mean classifier probability on the selected test prompts, capped at 100 by default | Not CTRAP's HS(IO)/HS(O) threshold rates or SDD's judge-based benchmark. A common comparison is legitimate when reported with its actual protocol. |

The local papers support these scope distinctions: SEAM specifies its 4k/4k/4k setup and 500-step defense; SDD specifies irrelevant-answer SFT, 8k balanced data, and 500 steps/batch 24/LR `5e-7`; CTRAP specifies LoRA and its original attack/metric setup. A single-seed shared sweep cannot reproduce the papers' full experimental tables or SEAM's 20-seed mean/std.

## Operational and repository observations

- SDD preflight validates seed/pool, file hashes, and pair counts before requesting a GPU. This is useful and materially stronger than checking only downloaded status text.
- Once SDD changes `status.json` from `data_preflight` to a full/train-push status, `_validate_preflight` will no longer accept the same preparation. A failed training run therefore cannot simply reuse that run ID through the normal launcher. Keep immutable preparation status separate from execution status.
- Full sweeps save individual metric files as cases complete, but write `summary.json` only at the end. There is no general resume path for an interrupted sweep. SDD training saves no intermediate training checkpoint. Preserve partial results and add stage-aware recovery before long reruns.
- The private HF upload helper verifies step count and refuses nonempty repos during preflight. Its reservation marker prevents ordinary repeated preflight, while upload-only retry paths use the existing checkpoint. Upload verification checks filenames, not remote weight hashes or a full reload; finiteness is checked for smoke training, but full-run upload validation does not reject nonfinite training metrics.
- Root `requirements.txt` omits `wandb`, although SEAM train/eval import it. Modal images add it separately, so this is a local README setup gap, not a demonstrated Modal image failure. Local Python is 3.13.5, Torch 2.9.1 CPU, Transformers 4.57.6; it is not the pinned GPU environment, and `datasets` is unavailable locally.
- `SEAM_MAIN_RESULT.md` describes an earlier snapshot: its claim that eval does not pass the attack flag is already fixed in the current source. The historical document should be marked as such.
- The generic `scripts/train.sh` uses 8,000 defense examples, unlike the current 4,000-example full Modal launcher. `scripts/eval.sh` still contains a placeholder model path and no explicit disjoint split. Prefer the documented comparison launchers over these legacy examples.
- Vendored SDD/CTRAP utilities have placeholders/local-path assumptions. Examples include SDD's empty model/input paths, its legacy OpenAI judge API, CTRAP's local token/data files, and README directory names that differ from the actual tree. The active adapters bypass several of these assumptions. They should not be presented as interchangeable executable entrypoints.
- `scripts/modal_seam_main.py` is the historical 20-step/smoke helper, with fixed output names and a missing training `--seed` override for non-42 seeds. It is not the 500-step full launcher.
- The checked local SDD status records one failed defense run and a separate completed 8k preparation. These are cached files, not a fresh confirmation of current remote state. The local 20-step SEAM loss files show finite values and increased harmful, refusal, and benign held-out losses; they do not establish HS/ZS robustness.

## Validation performed and limits

- All 44 Python files parsed successfully.
- All 11 PowerShell scripts parsed successfully.
- `python -m unittest discover -s scripts -p 'test_*.py' -v`: **8 tests passed**. They cover split construction on unique fixtures, SDD matching/preflight, sweep invocation construction, and HF upload guards.
- Additional small probes executed the actual extracted dataset tokenizer and SEAM loss methods, plus the CLI parser and async refusal verdict parser. Findings 1, 2, 4, 7, and 10 were reproduced locally.
- Relevant paper pages were extracted and visually inspected. Pinned upstream Trainer/collator and Harness source were checked for the API behavior described above.
- No CUDA training, actual dataset overlap audit, current Modal status check, model reload from HF, or end-to-end baseline evaluation was performed. This review does not certify the final remote checkpoints.
- Detailed semantic review focused on the active SDD/CTRAP/SEAM rerun paths. Repository inventory and syntax checks also cover vendored auxiliary scripts; this is not exhaustive runtime testing of every unused upstream baseline or downstream task.

## Recommended order for the ongoing reruns

1. Preserve existing run IDs, manifests, logs, and original defended checkpoints. Label their actual code/protocol; do not overwrite them.
2. Correct attack padding labels and harmfulness test construction, then run one controlled base attack and verify selected test IDs, trainable labels, generated outputs, and finite losses.
3. Correct and validate SEAM masking and inspect finite-difference precision before producing a new SEAM defense checkpoint. Keep the old released-code variant as a separately named result if needed.
4. Centralize MC1/MC2, raw/normalized accuracy, zero-shot settings, generation config, revisions, and effective batch size; apply that same evaluation to all three methods.
5. Reuse usable SDD/CTRAP defense checkpoints for corrected evaluation where their defense training is unchanged. Rerun harmful fine-tuning from those original checkpoints when the attack objective changes. Add recovery and split/hash validation before expanding to more seeds.
