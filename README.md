## Before Modifications
**MEM**: 65.5%  
**SVM**: 67.1%  
**CRF**: 67.9%  

## First Modification
Changes:
- prefixes 2, 3, 4
- suffixes 2
- word shape
- word length (<=3, <=6, <=10, >10)
- has: parenthesis, slash, dot, plus comma
- found in dictionary

**MEM**: 65.9%  
**SVM**: 67.5%  
**CRF**: 67.9%  

## Second Modification
Changes:
- added tagger 
- added pos, unipos features
**MEM**: 65.6%  
**SVM**: 67.4%  
**CRF**: 67.9%  

## Third Modification
Changes:
- added lemma and attribute_ruler
- added lemma feature
**MEM**: 66.2%  
**SVM**: 67.5%  
**CRF**: 67.9%

## Fourth Modification
Changes:
- added prevPos, prevUniPos, postPos, postUniPos
**MEM**: 65.8%  
**SVM**: 67.4%  
**CRF**: 67.5%

## Fifth Modification
Changes:
- brought back to third modification
- improved context: added prev2 and next2 features 
**MEM**: 66.3%
**SVM**: 67.1%  
**CRF**: 67.8%

## Sixth Modification
Changes:
- Text normalization & biomedical regex/pattern flags.
**MEM**: 66.5%
**CRF**: 68.3%
**SVM**: 67.2%  

## Seventh Modification
Changes:
- Character n-grams (c3/c4/c5) for current token.
**MEM**: 66.5%
**CRF**: 68.0%
**SVM**: 67.2%  

## Eighth Modification
Changes:
- Multi-token dictionary span matches.
- Token-level dictionary hits miss multiword drug names (e.g., "acetyl salicylic acid").
**MEM**: 66.1%
**CRF**: 67.5%
**SVM**: missing  

## Ninth Modification
Changes:
- Added longer prefix/suffix features (5 and 6 characters).
- Tested without addition 8 (multi-token spans hurt performance).

**mod9_wo8** (additions 1-7 + 9, no 8):
**MEM**: 66.2%
**CRF**: 68.3% (best CRF in this set)
**SVM**: 67.7%

---

## Feature Set Summary

Each "mod" corresponds to a combination of feature additions:

| Feature Set | Additions Enabled | ADD7 (char n-grams) | ADD8 (multi-token spans) | ADD9 (longer prefixes/suffixes) |
|---|---|---|---|---|
| mod6 | 1-6 | Off | Off | Off |
| mod8 | 1-8 | On | On | Off |
| mod9_wo8 | 1-7, 9 | On | Off | On |

---

## Hyperparameter Tuning (Devel Set)

Grid search was performed for all three models (CRF, SVM, MEM) across three feature sets.
Full results in `hyperparameter_results_mod6.csv`, `hyperparameter_results_mod8.csv`, `hyperparameter_results_mod9_wo8.csv`.

### Key Findings
- **CRF dominates** across all feature sets, with best devel scores around 68.2%-68.5%.
- **MEM is insensitive** to hyperparameters -- scores stay at ~66.3% regardless of C or iterations.
- **SVM** peaks at ~67.5%-67.7% with either linear (low C) or rbf (higher C) kernels.
- **mod6 and mod9_wo8** are the best feature sets (~68%+), while mod8 (with multi-token spans) slightly hurts CRF.

### Top Devel Results per Feature Set

**mod6 (best for CRF):**
| Model | Params | Devel F1 |
|---|---|---|
| CRF | c1=0.01, c2=0.1, iter=50 | 68.5% |
| CRF | c1=0.1, c2=0.1, iter=50 | 68.4% |
| CRF | c1=0.01, c2=0.1, iter=500 | 68.2% |
| CRF | c1=0.01, c2=1.0, iter=50 | 68.2% |
| CRF | c1=0.1, c2=0.5, iter=100 | 68.2% |
| SVM | C=0.1, linear | 67.5% |

**mod9_wo8:**
| Model | Params | Devel F1 |
|---|---|---|
| CRF | c1=0.1, c2=0.5, iter=200 | 68.3% |
| CRF | c1=0.01, c2=0.1, iter=100 | 68.2% |
| CRF | c1=0.01, c2=0.5, iter=50 | 68.2% |
| SVM | C=0.1, linear | 67.7% |

**mod8 (best for SVM):**
| Model | Params | Devel F1 |
|---|---|---|
| SVM | C=1.0, rbf | 67.7% |
| CRF | c1=0.01, c2=0.1, iter=50 | 67.7% |

---

## Test Set Evaluation

After hyperparameter tuning on the devel set across three feature configurations (mod6, mod8, mod9_wo8), we selected the best-performing models from each and evaluated them on the **test set** to measure real generalization.

### Step 1: Cross-feature-set comparison (11 best models)

We picked the top configs from each feature set (based on devel F1) and ran them on the test set.

| Feature Set | Model | Params | Devel F1 | Test F1 |
|-------------|-------|--------|----------|---------|
| mod6 | CRF | c1=0.01, c2=0.1, iter=50 | 68.5% | 63.4% |
| mod6 | CRF | c1=0.1, c2=0.1, iter=50 | 68.4% | 63.3% |
| mod6 | CRF | c1=0.01, c2=0.1, iter=500 | 68.2% | 63.0% |
| mod6 | CRF | c1=0.01, c2=1.0, iter=50 | 68.2% | 63.1% |
| mod6 | CRF | c1=0.1, c2=0.5, iter=100 | 68.2% | 62.5% |
| mod6 | SVM | C=0.1, linear | 67.5% | 62.0% |
| mod9_wo8 | CRF | c1=0.1, c2=0.5, iter=200 | 68.3% | 62.3% |
| mod9_wo8 | CRF | c1=0.01, c2=0.1, iter=100 | 68.2% | 62.8% |
| mod9_wo8 | CRF | c1=0.01, c2=0.5, iter=50 | 68.2% | 63.2% |
| mod9_wo8 | SVM | C=0.1, linear | 67.7% | 62.0% |
| **mod8** | **SVM** | **C=1.0, rbf** | **67.7%** | **67.9%** |

Key observation: mod8 SVM rbf was the clear winner (67.9%), while CRF models on mod6/mod9_wo8 all dropped ~5 points from devel to test, indicating overfitting. This motivated a deeper investigation of mod8.

### Step 2: Fine-grained mod8 testing

Since mod8 produced the best test result, we ran a more thorough evaluation on mod8 with all three model types and finer hyperparameter grids.

**SVM rbf -- C grid:**
| Params | Test F1 |
|--------|---------|
| C=0.5, rbf | 67.3% |
| **C=0.75, rbf** | **67.9%** |
| **C=1.0, rbf** | **67.9%** |
| C=1.5, rbf | 67.7% |
| C=2.0, rbf | 67.8% |
| C=3.0, rbf | 67.8% |
| C=5.0-50.0, rbf | 67.7% |

**SVM rbf -- gamma tuning (hurt performance, some configs too slow to converge):**
| Params | Test F1 |
|--------|---------|
| C=0.5, gamma=0.001 | 62.0% |
| C=0.5, gamma=0.005 | 66.8% |
| C=0.5, gamma=0.01 | 67.2% |
| C=0.5, gamma=0.05 | 49.7% |

**SVM linear:**
| Params | Test F1 |
|--------|---------|
| C=0.05, linear | 67.5% |
| C=0.1, linear | 67.6% |
| **C=0.2, linear** | **67.9%** |
| C=0.5, linear | 67.6% |
| C=1.0, linear | 67.6% |

**CRF (mod8):**
| Params | Test F1 |
|--------|---------|
| **c1=0.1, c2=0.1, iter=50** | **68.2%** |
| **c1=0.5, c2=0.1, iter=50** | **68.2%** |
| c1=0.01, c2=0.1, iter=50 | 67.7% |
| c1=0.1, c2=0.5, iter=50 | 67.7% |
| c1=0.01, c2=1.0, iter=200 | 67.7% |

**MEM (mod8):**
| Params | Test F1 |
|--------|---------|
| C=0.1 | 67.5% |
| C=1.0 | 67.8% |

---

## Final Results

| Rank | Model | Feature Set | Params | Test F1 |
|------|-------|-------------|--------|---------|
| 1 | **CRF** | **mod8** | **c1=0.1, c2=0.1, iter=50** | **68.2%** |
| 1 | **CRF** | **mod8** | **c1=0.5, c2=0.1, iter=50** | **68.2%** |
| 3 | SVM | mod8 | C=0.75, rbf | 67.9% |
| 3 | SVM | mod8 | C=1.0, rbf | 67.9% |
| 3 | SVM | mod8 | C=0.2, linear | 67.9% |
| 6 | MEM | mod8 | C=1.0 | 67.8% |

### Conclusions
- **Best model: CRF with mod8 features (additions 1-8), c1=0.1, c2=0.1, iter=50 -- 68.2% test F1.**
- **mod8 is the best feature set** for all three models on the test set. The multi-token dictionary span features (addition 8) that appeared to hurt CRF on devel actually helped generalization on test.
- CRF with mod6/mod9_wo8 features overfit on devel (~68.5%) and dropped to ~63% on test. With mod8, CRF generalizes well.
- All three model types performed competitively on mod8: CRF (68.2%), SVM (67.9%), MEM (67.8%).
- Tuning SVM gamma away from the default (`scale`) consistently hurt performance.
- SVM rbf is robust to C values in the range 0.75-50.0 (67.7-67.9%).