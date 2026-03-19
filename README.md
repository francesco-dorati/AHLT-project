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
**MEM**: 

---


 ## Suggested Temporal Workflow
 - Baseline Verification: Run the provided system with the current features to establish your starting score (approx. 65%-67%).
 - Linguistic Features (The Foundation): Add PoS Tags and Lemmas. These are fundamental to NLP and usually provide the biggest initial boost.
 - Context Expansion: Implement the $\pm 2$ window. This adds "breadth" to your model's vision.
 - Domain-Specific Features: Add regex for chemical symbols, dashes, and digits.
 - Algorithm Comparison: Once your features are "frozen," train all three models (MEM, SVM, CRF) on the same feature set.Hyperparameter Tuning: Pick your best algorithm (likely CRF) and fine-tune its $C$ parameter.