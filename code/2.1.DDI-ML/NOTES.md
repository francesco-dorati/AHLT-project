# System 2.1 — DDI with Machine Learning — Experiment Log

This file tracks every experiment we run for System 2.1 so we can write the
report straight from this log when the time comes.

## Conventions

- **Code modifications** are tagged in source with `[MOD-2.1]` comments
  (same convention as Task 1, replacing `[MOD-1.x]`). Always include a one-line
  explanation of *why* the change was made next to the tag.
- **Experiment names** follow the pattern `<algo>-<feat-set>-<hp>` where:
  - `<algo>` ∈ {MEM, SVM}
  - `<feat-set>` ∈ {ref, mod1, mod2, …} (ref = shipped baseline features)
  - `<hp>` is a compact hyperparameter suffix, e.g. `C1_lbfgs1500`
- **Output files** live in `code/2.1.DDI-ML/results/` as
  `devel-<expname>.{out,stats}` and `test-<expname>.{out,stats}`.
- **Feature dumps** live in `code/2.1.DDI-ML/preprocessed/` as
  `train.feat` and `devel.feat`. Re-extraction needed whenever
  `extract_features.py` or `patterns.py` change.
- **Models** live in `code/2.1.DDI-ML/models/` as `model.MEM` / `model.SVM`.
- Every experiment is logged in the table below the moment it finishes.

## Reference points (DDI is much harder than NER, expect low numbers)

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based "words in between" | re-run locally 2026-05-02 | 13.1% | 22.2% | 20.8% | 26.9% |
| 2.1 reference ML (provided extract_features) | spec p35 | TBD | TBD | TBD | TBD |
| Ballpark "good" ML target | spec p35 | — | ~65% M | — | — |

### 2.0 baseline per-class breakdown (matches spec exactly):

```
devel:   advise 0.0  effect 12.3  int 70.7  mechanism 5.8   | M=22.2  m=13.1
test:    advise 20.1 effect 25.2  int 50.0  mechanism 12.3  | M=26.9  m=20.8
```

The rule-based baseline relies on ~50 hand-curated cue words split by class.
It nails `int` (the rarest class — 43 devel pairs) because the cue list
includes `interact`, `tylenol`, `mivacron`. It collapses on `advise` (0% F1
on devel) because the cue list for `advise` consists of overly-specific
drug-name tokens like `dihydroergotamine`, `cyp2d6`, `narrow` etc.

Note: DDI evaluator reports m.avg (micro) and M.avg (macro) across the 4
positive classes (`advise`, `effect`, `int`, `mechanism`). `null` pairs
are excluded from F1 by construction.

## Experiment table

| Run | Algorithm | Features | Hyperparameters | devel m | devel M | test m | test M | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| baseline-2.0 | rule | cue words | n/a | 13.1 | 22.2 | 20.8 | 26.9 | re-run 2026-05-02 |
| ref-MEM | MEM | shipped | C=1 lbfgs maxit=1500 | — | — | — | — | running |
| ref-SVM | SVM | shipped | C=1 rbf gamma=scale | — | — | — | — | pending after MEM |

## Feature-set catalogue

| Tag | Description | Source |
|---|---|---|
| `ref` | Shipped feature set (type, samedrug, LCS lemma+pos, 9 path variants, words/lemmas in path, lcs children, 4 patterns: verb-lcs / verb-func / wib / wout) | `extract_features.py` + `patterns.py` shipped |

Add new rows under "Feature-set catalogue" whenever a new feature mod is
introduced; reference the mod tag in the run name (e.g. `MEM-mod1-…`).

## Algorithm sweep plan (provisional)

| Algorithm | Hyperparameter | Values to try |
|---|---|---|
| MEM | C | 0.1, 0.5, 1.0, 5.0, 10.0 |
| MEM | solver | lbfgs, liblinear, saga |
| MEM | max_iter | 500, 1500, 5000 |
| SVM | C | 0.1, 0.5, 1.0, 5.0, 10.0 |
| SVM | kernel | linear, rbf, poly |
| SVM | degree (poly only) | 2, 3 |
| SVM | gamma | scale, auto, 0.01, 0.1 |

Feature engineering takes priority over hyperparameter tuning. Hyperparam
sweep only happens after we lock in the best feature mod, mirroring the
Task 1 1.1 (CRF) methodology.

## Feature engineering ideas (from spec p41–46)

- **Position features**: word lemmas/POS before E1, between E1/E2, after E2
- **Clue verbs**: presence + position of pharmacology-trigger verbs
  (interact, inhibit, induce, potentiate, antagonize, increase, decrease, affect, alter, …)
- **Third entity in path / sentence**: helps disambiguate when 3+ drugs co-occur
- **Entity-pair type combinations**: `typeE1+typeE2` as a combined feature
- **More tree-pattern features**: LCS verb subtree shape, subject/object roles
- **Negation cues**: presence of "not", "no", "without" near the verb
- **Sentence length / distance** between entities (bucketed: <5, 5-10, 10-20, 20+)

These are candidates only; we'll prioritise based on baseline error analysis.

## Open questions / decisions to log

- Whether to use spaCy's `en_core_web_trf` (transformer, slow but accurate)
  or `en_core_web_sm` (small, fast). Shipped uses `trf`. We'll stick with
  `trf` for the report; document any swap.
- Whether to evaluate the rule-based baseline 2.0 once and quote it from
  the spec, or include it in our cross-system comparison row. Decision:
  include it (re-run to confirm).
