# System 2.2 — DDI with Neural Networks — Experiment Log

This file tracks every experiment for System 2.2 (CNN/LSTM hybrid DDI
classifier). Same convention as System 1.2 / 2.1 — every code change
tagged `[MOD-2.2]` with a one-line reason.

## Reference points

| System | Source | devel m-F1 | devel M-F1 | test m-F1 | test M-F1 |
|---|---|---:|---:|---:|---:|
| 2.0 rule-based | 2.0 re-run | 13.1 | 22.2 | 20.8 | 26.9 |
| **2.1 ML champion** (two-stage on mod_best2, t=0.37) | 2.1 NOTES.md | 65.3 | 65.9 | 62.5 | 66.8 |
| 2.2 ref (shipped CNN+BiLSTM) | TBD | — | — | — | — |

2.2 spec gives no specific target number, just says ≥ 90 % validation
accuracy yields a "reasonable F₁" — accuracy is misleading because ~85 %
of pairs are `null` (predicting all-null already hits 85 % acc with
F₁=0).

## Shipped architecture (network.py:ddiCNN)

```
Input: 3 indexed sequences of length max_len (default 150):
  - lc_form  (lowercased word)  -> embedding 100
  - lemma                       -> embedding 100
  - pos                         -> embedding 50
Concat -> 250
BiLSTM(250 -> 200) + Dropout(0.2)
MaxPool1d(kernel=4, stride=1, padding=1)
Conv1d(200 -> 64, kernel=2, padding=same) + ReLU
MaxPool1d(kernel=max_len-1)
Flatten + Dropout(0.2)
Linear(64*max_len -> 5)
```

Despite being called "CNN" in the spec, it's a BiLSTM-then-CNN hybrid.

## Conventions

- `[MOD-2.2]` tag in source for every code change.
- Models stored in `code/2.2.DDI-NN/models/<name>/`.
- Results in `code/2.2.DDI-NN/results/{devel,test}-<name>.{out,stats}`.
- Parsed pickles in `code/2.2.DDI-NN/preprocessed/{train,devel,test}.pck`.
- Each run named `<arch_tag>_bs<BS>_ml<ML>_ep<EP>` or similar.
- Training uses a fixed seed (2345) for reproducibility (train.py sets it).

## Experiment table

| Run | Architecture | HP | devel m | devel M | test m | test M | Notes |
|---|---|---|---:|---:|---:|---:|---|
| ref_bs16_ml150_ep10 | CNN+BiLSTM shipped | bs=16 ml=150 ep=10 seed=2345 | 59.4 | 55.8 | — | — | val-acc 88.35% (≈ spec target); below 2.1's M=65.9 because NN lacks dep-tree features |

### ref baseline per-class (devel)

```
                   P     R    F1
advise           68.0  69.9  69.0
effect           66.2  60.1  63.0
int              48.4  34.9  40.5   ← much worse than 2.1's 68.4
mechanism        74.1  38.3  50.5   ← worse than 2.1's 61.6
M.avg            64.2  50.8  55.8
m.avg            67.5  53.1  59.4
```

**Observation**: NN baseline trails 2.1-ML on M-F1 by **-10 pp** because the
shipped CNN+BiLSTM has only `lc_form/lemma/pos` inputs — no dependency-
tree features (which carry most of the DDI signal in 2.1). This is the
main opportunity: enrich inputs (suffix, prefix, entity-type marker)
and consider syntactic features.

## Campaign plan

### Phase R — Reference (shipped architecture, default HPs)
Run the shipped `ddiCNN` with bs=16, ml=150, ep=10. Establishes the starting
point.

### Phase I — Input representations
- mod1: add case-sensitive `form` input (4th embedding stream)
- mod2: add suffix-of-form input (codemaps already accept `suf_len` param
  but doesn't use it — wire it up)
- mod3: add prefix-of-form input
- mod4: add entity-type indicator embedding (whether each token is
  DRUG1/DRUG2/DRUG_OTHER/other)
- mod5: pretrained embeddings (spaCy `en_core_web_md` or GloVe-100d)

### Phase A — Architecture
- arch1: pure CNN (drop the BiLSTM)
- arch2: pure BiLSTM (drop the CNN)
- arch3: 2-layer BiLSTM
- arch4: bigger embeddings (200 instead of 100)
- arch5: bigger LSTM hidden (200 instead of 100)
- arch6: add attention over LSTM outputs
- arch7: deeper CNN (3 conv layers, multi-kernel)

### Phase H — Hyperparameters (on best arch from A)
- HP1: batch sizes {8, 16, 32, 64}
- HP2: epochs {5, 10, 20, 30}
- HP3: dropout rates {0.1, 0.2, 0.3, 0.5}
- HP4: max_len {100, 150, 200}
- HP5: learning rate (need to surface as param, currently hardcoded)

### Phase S — Seed audit
3-seed audit on the champion arch+HP. Same methodology as System 1.2's
Round 5 — single-seed peaks can be lucky.

### Phase T — Final test eval
Run the devel-selected champion on test once.

## Open questions / decisions to log

- The shipped `Codemaps` doesn't actually use `suf_len` even though
  train.py forwards it. We'll wire it up as mod2.
- Loss is `nn.CrossEntropyLoss()` applied to argmax-encoded labels
  (one-hot). Standard for multi-class — no change needed.
- Optimizer is `Adam` with default params (lr=1e-3 implicit).
- LSTM training on CPU is slow (no local GPU). Each training run
  estimated ~5-15 min depending on epochs. If too slow we can move to
  Boada (same as 1.2 / 1.3 workflow).
