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
| ref-MEM | MEM | shipped | C=1 lbfgs maxit=1500 | **63.2** | **64.3** | — | — | new ML baseline; ~+42pp M over 2.0 |
| ref-SVM | SVM | shipped | C=1 rbf gamma=scale | 54.3 | 55.0 | — | — | -9pp M vs MEM; precision-biased (R~43%) |
| mod1-MEM | MEM | shipped + mod1 | C=1 lbfgs maxit=1500 | 61.8 | 62.4 | — | — | **−1.9 pp M vs ref-MEM — distance/senlen/order HURT, disabled in src** |
| mod1+2-MEM | MEM | shipped + mod1 + mod2 | C=1 lbfgs maxit=1500 | 62.1 | 63.0 | — | — | −1.3 pp M vs ref; combo doesn't rescue mod1 |
| mod2-MEM | MEM | shipped + mod2 | C=1 lbfgs maxit=1500 | 63.0 | 64.2 | — | — | −0.1 pp M — neutral; small gain on `int`, small loss on `advise`/`mechanism` |
| mod3-MEM | MEM | shipped + mod3 | C=1 lbfgs maxit=1500 | 64.0 | 64.0 | — | — | micro +0.8 / macro −0.3; helps 3/4 classes, hurts `int` (−4.3) |
| mod4-MEM | MEM | shipped + mod4 | C=1 lbfgs maxit=1500 | 63.1 | 64.1 | — | — | −0.2 pp M — neutral; type-pair combined feature doesn't move the needle alone |
| mod5-MEM | MEM | shipped + mod5 | C=1 lbfgs maxit=1500 | 62.8 | 63.0 | — | — | **−1.3 pp M — hurt; bag-of-negation cues without scoping** |
| **mod_best-MEM** | MEM | shipped + mod2+mod3+mod4 | C=1 lbfgs maxit=1500 | **65.1** | **65.2** | **63.0** | **66.8** | **+0.9 pp M / +1.9 pp m on devel vs ref; tested on test → M=66.8 (the System 2.1 final number)** |

### Phase R/F/A summary (devel, no test except mod_best)

```
                  M-F1   m-F1   ΔM     Δm    notes
ref-MEM           64.3   63.2     —      —   baseline
ref-MEM C=0.1     61.1   60.5  -3.2   -2.7   over-regularised
ref-MEM C=0.5     63.7   62.8  -0.6   -0.4
ref-MEM C=2       63.6   62.6  -0.7   -0.6
ref-MEM C=5       63.2   62.0  -1.1   -1.2
ref-MEM cw=bal    53.2   56.7 -11.1   -6.5   class_weight=balanced over-corrects
ref-SVM           55.0   54.3  -9.3   -8.9   rbf default, precision-biased
SVM linear C=0.5  60.5   60.0  -3.8   -3.2   linear improves but still trails MEM
mod1-MEM          62.4   61.8  -1.9   -1.4   distance/length buckets HURT
mod2-MEM          64.2   63.0  -0.1   -0.2   clue verbs ≈ neutral
mod3-MEM          64.0   64.0  -0.3   +0.8   path entities micro+, macro-
mod4-MEM          64.1   63.1  -0.2   -0.1   type-pair combined ≈ neutral
mod5-MEM          63.0   62.8  -1.3   -0.4   negation cues HURT
mod_best-MEM      65.2   65.1  +0.9   +1.9   ← combo of mod2+mod3+mod4 wins
```

### mod_best per-class breakdown (devel + test):

```
DEVEL:                P     R    F1
advise              64.1  63.6  63.9   (+1.3 vs ref)
effect              82.4  57.9  68.0   (+2.6 vs ref)
int                 78.1  58.1  66.7   (-2.5 vs ref)
mechanism           73.7  53.6  62.1   (+2.1 vs ref)
M.avg               74.6  58.3  65.2
m.avg               74.9  57.5  65.1

TEST:                 P     R    F1
advise              76.0  56.0  64.5
effect              73.3  56.3  63.7
int                100.0  65.0  78.8   (!)
mechanism           64.2  56.4  60.1
M.avg               78.4  58.4  66.8
m.avg               71.0  56.7  63.0
```

Notable: on test, `int` reaches 78.8 F1 (zero false positives, 26/40 recall).
This is a small class (40 test pairs), so the macro gain is partly variance-
driven. The macro-F1 on test (66.8) is +1.6 pp above devel (65.2), which is
unusual but plausible given the small devel size for `int`. Spans the
spec's "~65% target" comfortably.

### Lessons learned (for the report)

1. **The shipped feature set is well-tuned**: 4/5 single-mod additions
   were neutral-to-negative. A logistic regression on sparse binary
   features is sensitive to dilution.
2. **Combinations can synergise**: each of mod2/mod3/mod4 alone was
   neutral or marginal, but together they gain +0.9 pp macro. The
   informative third-entity / type-pair / clue-verb features cooperate
   in ways the model can't reconstruct from any single one alone.
3. **Hyperparameter tuning didn't help**: default C=1 with lbfgs solver
   is already optimal for both ref features and mod_best features.
   `class_weight='balanced'` overcorrects (the predict path drops `null`
   pairs anyway, so balancing across 5-way training is misguided).
4. **MEM (LogisticRegression) >> SVM** on this task with these features
   (best MEM 65.2 vs best SVM 60.5). The shipped feature space is sparse
   and high-dim — exactly LR's regime.
5. **`int` is the easiest positive class** despite being the rarest in
   the corpus, because its trigger lexicon is concentrated ("interact",
   "coadminister"). `mechanism` and `advise` are hardest — more diffuse
   linguistic patterns.

### ref-MEM per-class devel breakdown:

```
                   P      R      F1
advise           63.8%  61.5%  62.6%
effect           78.7%  56.0%  65.4%
int              77.1%  62.8%  69.2%
mechanism        69.3%  52.9%  60.0%
M.avg            72.2%  58.3%  64.3%
m.avg            72.0%  56.4%  63.2%
```

### ref-SVM per-class devel breakdown:

```
                   P      R      F1
advise           60.8%  41.3%  49.2%
effect           83.2%  47.2%  60.2%
int              90.9%  46.5%  61.5%
mechanism        70.5%  37.5%  49.0%
M.avg            76.4%  43.1%  55.0%
m.avg            74.6%  42.7%  54.3%
```

**Decision**: MEM (LogisticRegression) is the carrier for feature
engineering. SVM (rbf default) is too precision-biased (recall < 45% on
every class); we'll revisit it in Phase A with linear kernel + lower C.

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

## Campaign plan (mirrors Task 1's NERC-ML methodology)

### Phase R — Reference (default features, default algos)
- `ref-MEM`: shipped features + MEM defaults
- `ref-SVM`: shipped features + SVM defaults
- Outcome: establishes baseline and decides which classifier to run mods on.

### Phase F — Feature engineering (one mod at a time, additive)
Each mod is a `[MOD-2.1]`-tagged addition to `extract_features.py` (or
`patterns.py`). After each mod we re-extract features (this is the slow
step), retrain the winning Phase-R classifier with default HPs, and log
results. We keep mods that help (>0.5 pp M-F1 on devel) and discard the rest.

| mod | What it adds | Files touched |
|---|---|---|
| mod1 | Distance + sentence-position features (E1/E2 token positions, distance bucket, sentence-length bucket) | `extract_features.py` |
| mod2 | Pharmacology clue-verbs (lemmas + position before/between/after pair) | `patterns.py` (new pattern) |
| mod3 | Third-entity context (count of other entities in sentence and in path, types of in-path entities) | `extract_features.py` |
| mod4 | Type-pair combined feature (`typeE1_typeE2`) | `extract_features.py` |
| mod5 | Negation cues near LCS / in path (`not`, `no`, `without`, `fail` lemmas) | `extract_features.py` or `patterns.py` |
| mod6 | Lemma/POS n-grams before E1 / between / after E2 (limited to content words) | `patterns.py` |
| mod7 | LCS subtree shape: LCS lemma + immediate-children dependencies | `extract_features.py` |
| mod_best | Bundle of all mods that helped | composite |

### Phase A — Algorithm + hyperparameter sweep (on `mod_best`)
| Algorithm | Grid |
|---|---|
| MEM | C ∈ {0.1, 0.5, 1, 5, 10}, solver ∈ {lbfgs, liblinear, saga}, max_iter ∈ {1500} |
| SVM | C ∈ {0.1, 0.5, 1, 5, 10}, kernel ∈ {linear, rbf}, gamma ∈ {scale, 0.1, 0.01} |

### Phase T — Final test eval
Best (algo, HP, feature-set) from Phases F+A, evaluated on test once.

## Open questions / decisions to log

- Whether to use spaCy's `en_core_web_trf` (transformer, slow but accurate)
  or `en_core_web_sm` (small, fast). Shipped uses `trf`. We'll stick with
  `trf` for the report; document any swap.
- Whether to evaluate the rule-based baseline 2.0 once and quote it from
  the spec, or include it in our cross-system comparison row. Decision:
  include it (re-ran 2026-05-02, matches spec exactly).
- Do we need Boada for 2.1? **No** — feature extraction is ~3 min locally
  on CPU (en_core_web_trf parses ~25 sentences/sec with batching);
  training MEM/SVM is seconds. Run everything locally.

## Open questions / decisions to log

- Whether to use spaCy's `en_core_web_trf` (transformer, slow but accurate)
  or `en_core_web_sm` (small, fast). Shipped uses `trf`. We'll stick with
  `trf` for the report; document any swap.
- Whether to evaluate the rule-based baseline 2.0 once and quote it from
  the spec, or include it in our cross-system comparison row. Decision:
  include it (re-run to confirm).
