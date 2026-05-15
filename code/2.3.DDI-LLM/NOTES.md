# System 2.3 — DDI with LLMs — Experiment Log

Same convention as System 1.3 / 2.1 / 2.2 — every code change tagged
`[MOD-2.3]` with a one-line reason. Mirrors the 1.3 NER-LLM campaign:
few-shot sweep → fine-tuning → final test eval.

## Reference points

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based "wib" | 2.0 re-run | 13.1 | 22.2 | 20.8 | 26.9 |
| 2.1 ML champion (two-stage on mod_best2) | 2.1 NOTES.md | 65.3 | 65.9 | 62.5 | **66.8** |
| 2.2 NN champion (mod9 rel-pos, seed 777) | 2.2 NOTES.md | — | 64.7 | 62.7 | **65.6** |
| 2.3 LLM ref (few-shot 0-shot baseline) | TBD | — | — | — | — |

DDI is sentence classification (5 classes incl. null). The LLM emits
a single token (the class name) per (sentence, drug pair).

## Task summary

- Each test instance = a sentence with `[DRUG1]` / `[DRUG2]` / `[DRUG_OTHER]` masks
- Possible answers: `mechanism`, `effect`, `advise`, `int`, or `null`
- ~85% of pairs are `null` (gold class distribution heavily skewed)
- Output format: single word (no XML, no offset reconstruction needed)
  — unlike 1.3 NER, there is **no offset-shift bug** to worry about
- Evaluator: `util/evaluator.py DDI`

## Provided infrastructure

```
code/2.3.DDI-LLM/bin/
├── examples.py          # Examples loader (DDI task)
├── prompts.py           # Prompts loader (same code as 1.3 — keep)
├── prompts01.json       # initial prompt (sysprompt + usrprompt)
├── fewshot.py           # few-shot inference driver
├── fewshot.sh           # SBATCH wrapper: 48 GB / RTX 3080
├── finetune-train.py    # LoRA fine-tuning trainer
├── finetune-inference.py# inference using saved LoRA adapter
├── FT-train.sh          # SBATCH: 64 GB / RTX 4090
├── FT-inference.sh      # SBATCH: 48 GB / RTX 3080
├── model.py             # Inference + FineTuning HF wrappers (same as 1.3)
└── paths.py
```

## Known issues from 1.3 transferred to 2.3

1. **`prompts.py` instruction-repetition pattern** — same code as 1.3.
   The prof's 2026-04-24 announcement says the *structure* is correct
   (usrprompt is repeated per-shot, preserving chat-template alternation).
   The fix is to keep `usrprompt` short — already mostly true in 2.3's
   prompts01.json (~6 lines, shorter than 1.3's prompts01).

2. **No offset bug** for DDI — `DDI_eval_format` just emits the class name,
   no XML tag offsets to manipulate.

## Spotted potential issue: `advice` vs `advise`

`prompts01.json` sysprompt enumerates classes as `mechanism`, `effect`,
**`advice`**, `int` — but the DDI-2013 dataset uses **`advise`** (no `c`).
The evaluator and the gold labels expect `advise`. Few-shot examples
(when given) will teach `advise` via the assistant turns, so the model
likely picks up the right spelling — but the system instruction nudges it
toward `advice`. **Worth testing: fix the typo and see if it helps.**

## SBATCH right-sizing (per prof's 2026-05-14 email)

- Inference jobs (fewshot.sh, FT-inference.sh): **48 GB** seems within
  guidance for LLM tasks. Probably fine.
- Training jobs (FT-train.sh): currently **64 GB**. Prof's note says
  oversizing hurts queue throughput. LoRA fine-tune of 3B-quantized
  models on the cluster has previously fit in ~40–48 GB. Will trim
  before launching large sweeps if 48 GB confirmed sufficient.
- **Maintenance window: 2026-06-03 08:00-17:00.** Avoid scheduling jobs
  that would span this.

## Conventions

- `[MOD-2.3]` tag in source for every code change.
- Models stored on Boada at `/scratch/nas/1/PDI/mml0/models/{name}` or
  via ollama for inference-only runs.
- Results in `code/2.3.DDI-LLM/results/{FS,FT}-<model>-<config>.{json,out,stats}`.

## Campaign plan (mirrors 1.3)

### Phase C — Few-shot sweep
- Phase C1: baseline = 0-shot llama32B3 with shipped prompts01
- Phase C2: shots sweep {0, 3, 5, 10, 15} on prompts01
- Phase C3: prompts × shots — try refined prompt variants (prompts02, prompts03)
- Phase C4: model swap (qwen25B3 vs llama32B3) at the best shots/prompt
- Phase C5: balanced vs unbalanced few-shot sampler

### Phase D — Fine-tuning baseline
- LoRA r=8, 10 epochs, lr=2e-5, 4-bit base, prompts01

### Phase F — FT sweep
- r=32 (bigger rank), more epochs, different models

### Phase H — best system consolidation (3-seed audit?)
- Multi-seed FT to check the lucky-seed pattern

### Phase I — final test eval

### Phase T — final report numbers

## Experiment table

| Run | Type | Model | Prompts | Shots | Balanced | Quant | devel m | devel M | test m | test M | Notes |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| _baseline_ | FS | llama32B3 | prompts01 | 0 | — | yes | — | — | — | — | TBD |
