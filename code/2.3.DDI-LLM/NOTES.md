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

### Phase C — Few-shot sweep (devel)

| Run | Model | Prompts | Shots | Balanced | devel M | devel m | Notes |
|---|---|---|---|---|---:|---:|---|
| C1.0 | llama32B3 | prompts01 | 0 | — | 9.5 | 8.2 | 0-shot collapses |
| C1.3 | llama32B3 | prompts01 | 3 | yes | 21.4 | 18.7 | huge FP load on `int` (1866 FPs) |
| C1.5 | llama32B3 | prompts01 | 5 | yes | 21.0 | 18.7 | |
| **C1.10** | **llama32B3** | **prompts01** | **10** | **yes** | **23.2** | 17.4 | **best macro of FS sweep** |
| C1.15 | llama32B3 | prompts01 | 15 | yes | 21.8 | 22.0 | best m |
| C2.5 | llama32B3 | prompts02 | 5 | yes | 20.7 | 23.3 | best micro overall |
| C2.10 | llama32B3 | prompts02 | 10 | yes | 21.8 | 23.0 | |
| C2.15 | llama32B3 | prompts02 | 15 | yes | 18.7 | 20.7 | |
| C4.10 | qwen25B3 | prompts01 | 10 | yes | 19.8 | 16.4 | qwen worse than llama |

### Phase C findings

- **Few-shot is fundamentally weak on DDI** at M=18-23. Compare to 2.1 ML
  (M=66.8 test) and 2.2 NN (M=65.6 test) — LLMs in FS mode trail by ~45 pp.
- **prompts02 (fixed "advise" + stronger null emphasis)** improved micro
  (~m+5 pp) but barely moved macro. The model is so confused on the
  positive/null boundary that prompt-level tweaks don't help much.
- **Adding shots beyond 5 helps only marginally** (3→10 shots: +1.8 pp;
  10→15 shots: -1.4 pp). The signal saturates because the model can't
  generalize the structural distinction "is there an interaction stated"
  from a handful of examples.
- **Qwen-2.5-3B is worse than Llama-3.2-3B** at 10-shot (19.8 vs 23.2).
  Same direction as 1.3 where qwen also trailed at few-shot.
- **Massive false-positive load**: model emits a positive class for ~85%
  of test pairs even though only ~15% are positive. Especially `int`
  (1-2% precision: ~1900 FPs for ~20 TPs). The instruction "if unsure,
  choose null" is being ignored.

**Implication**: fine-tuning is the only path to closing the gap to ML/NN.
Phase D is running now (job 425328, llama32B3 + prompts01 + 4-bit, LoRA r=8).

### File-naming fix (mid-campaign)

The shipped `fewshot.sh` named outputs as `FS-<model>-<shots>-<test><quant>.out`
without the prompt-variant tag — same bug as 1.3. So when we ran prompts02
at 5/10/15 shots, the .out/.stats files overwrote the matching prompts01
runs. Local copies of the prompts01 results were already saved.

Fix applied (mirroring 1.3):

```bash
TAG=$(basename "$PROMPTS" .json)
BASE=../results/FS-$MODEL-$SHOTS-${TEST}${QUANT}
TAGGED=../results/FS-$MODEL-$TAG-$SHOTS-${TEST}${QUANT}
mv $BASE.{out,json} $TAGGED.{out,json}
python3 evaluator.py DDI ... $TAGGED.out $TAGGED.stats
```

Overwritten files on Boada renamed to `FS-llama32B3-prompts02-{5,10,15}-…`.

## Phase D + F — Fine-tuning (final results)

Trained four LoRA fine-tune configurations on the train split (5000
balanced examples) with 10 epochs, lr=2e-5, batch=1, gradient_accum=8,
prompts01.json. All 4-bit quantised.

| Config | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| FT Llama r=8  | 28.9 | 36.0 | 30.5 | 35.3 |
| FT Qwen  r=8  | 33.7 | 33.7 | 27.6 | 29.4 |
| **FT Llama r=32** | **33.8** | **37.6** | **35.2** | **37.8** |
| FT Qwen  r=32 | 32.5 | 34.5 | 29.1 | 32.0 |

**Champion: FT Llama r=32 — devel M=33.8, test M=35.2.** Per-class on test:

```
                P     R    F1
advise        27.7  78.5  40.9
effect        27.3  67.9  38.9
int           17.6  40.0  24.4
mechanism     23.8  78.5  36.5
M.avg         24.1  66.2  35.2
m.avg         25.5  73.3  37.8
```

Pattern: **all classes have high recall (40-79%) but low precision (17-28%)**.
The fine-tuned LLM still over-predicts positives (the prompt's "null"
instruction is hard to obey under heavy class imbalance).

### Phase D/F findings

1. **Rank r=32 helps a lot.** Llama r=32 beats Llama r=8 by +4.7 pp test
   macro. Same direction as 1.3 NER campaign. The bigger adapter has
   more capacity for the multi-class boundaries.

2. **Llama > Qwen for DDI at this scale.** Llama r=32 wins both devel
   (33.8 vs 32.5) and test (35.2 vs 29.1). Same as in 1.3 (Llama r=32
   won test) but at a much lower absolute level.

3. **FT is much better than FS but still far below ML/NN.** FT Llama
   r=32 test M=35.2 vs FS best 23.2 → +12 pp. But ML champion is 66.8
   and NN champion is 65.6 — the LLM trails by ~30 pp.

4. **Why? DDI ≠ NER.** Unlike 1.3 where FT-LLM matched/beat dedicated
   NERC, here the LLM struggles because:
   - The class imbalance is extreme (85% null); the model can't reliably
     refuse to predict a class.
   - DDI requires *relational* reasoning between the two DRUG markers,
     not just lexical matching. A 3B-quantised LLM lacks the deep
     reasoning that the ML's syntax features and the NN's positional
     embeddings encode explicitly.
   - Output is a single token (the class name) — there's no
     "structured generation" gain, unlike NER where the LLM can stream
     XML tags.

5. **Disk quota mid-campaign caused silent failures.** Several
   FT-inference jobs ran for 13-21s producing empty .out files because
   the Boada 2GB user quota was exhausted by retained training
   checkpoints. Lesson: clean `checkpoint-*` subdirs aggressively after
   each FT-train completes.

## Cross-system comparison

| System | Devel M | Devel m | Test M | Test m |
|---|---:|---:|---:|---:|
| 2.0 rule baseline | 22.2 | 13.1 | 26.9 | 20.8 |
| 2.1 ML two-stage (mod_best2) | 65.9 | 65.3 | **66.8** | 62.5 |
| 2.2 NN mod9 rel-pos (seed 777) | 64.7 | 62.7 | **65.6** | 62.7 |
| 2.3 LLM FT-Llama r=32 (champion) | 33.8 | 37.6 | **35.2** | 37.8 |

For DDI:
- **ML > NN by ~1 pp** (66.8 vs 65.6 test macro)
- **LLM trails both by ~30 pp** at the 3B-quantised scale

This is the *opposite* of 1.3 NER where the LLM matched/beat the
dedicated systems. The key task differences explain it (see point 4
above).
