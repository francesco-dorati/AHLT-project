Here is a complete, professional, and detailed plain-text report of exactly what you built, tested, and achieved for System 1.2. You can use this as your final submission text, a README, or your personal notes for presenting your work.

***

# System 1.2: Named Entity Recognition with Neural Networks (Final Report)

## Executive Summary
The objective of System 1.2 was to build a deep learning architecture capable of outperforming traditional feature-based machine learning (CRF) for biomedical Named Entity Recognition. By engineering linguistic input features, integrating pre-trained semantic embeddings, applying aggressive regularization, and conducting automated hyperparameter tuning, the final neural network achieved a **69.8% Macro-F1** on the unseen test set, successfully beating the machine learning baseline.

---

## Step-by-Step Implementation & Methodology

### 1. Automated Training Infrastructure
To efficiently test different network configurations without manual intervention, I engineered a robust, crash-resistant Bash testing pipeline. 
* **Grid Search:** The script systematically looped through combinations of Batch Sizes (16, 32), Sequence Lengths (100, 150), Suffix Lengths (4, 5), and Epochs.
* **State Management:** It parsed the resulting evaluation stats in real-time and logged the Macro-F1 scores to a CSV file. If the script was interrupted, it automatically skipped previously successful runs, saving significant computational time.

### 2. Linguistic Feature Engineering (Syntax & Morphology)
The baseline neural network only embedded lowercased words and suffixes. To give the model a better understanding of biomedical text, I expanded the input tensors:
* **Part-of-Speech (PoS) Tags:** I used spaCy to extract coarse PoS tags for every token and mapped them into a dedicated embedding layer. This acted as a syntactic chunker, helping the network easily separate nouns from verbs and prepositions.
* **Prefix Embeddings:** I modified the data extraction pipeline to capture the first 4 characters of every word. This provided the network with crucial morphological clues about pharmacological naming conventions (e.g., *anti-*, *poly-*).

### 3. Semantic Upgrades via Pre-trained GloVe Embeddings
To give the model a prior understanding of English and medical terminology, I replaced the network's random weight initialization with Stanford's 100-dimensional GloVe embeddings.
* I wrote a custom vocabulary loader that successfully matched approximately 80% of the dataset's unique words to pre-calculated GloVe vectors. 
* This allowed the network to map semantically related drugs (like "aspirin" and "ibuprofen") to similar mathematical spaces before training even started.

### 4. Diagnosing and Fixing Severe Overfitting
While the GloVe embeddings provided a massive learning boost, they caused the model to overfit violently—memorizing the training data by Epoch 3 while validation scores crashed. To stabilize the network, I implemented an "Anti-Overfitting Combo" in the PyTorch architecture:
* **Deepened the Network:** Upgraded from a 1-layer to a 2-layer Bidirectional LSTM.
* **Heavy Dropout:** Increased the dropout rate across all embedding layers and the LSTM from 0.1 to 0.3.
* **Layer Normalization:** Added an `nn.LayerNorm` layer immediately following the LSTM to stabilize the gradients.
* **Advanced Activation:** Swapped the standard ReLU activation function for the smoother GELU function.

### 5. Final Hyperparameter Tuning
With the regularized GloVe architecture in place, I executed a final automated grid search. Because of the heavy dropout and LayerNorm, the model learned steadily rather than crashing early. 
* The model required a longer sequence length to capture extended context in complex medical sentences.
* **Development Peak:** The optimal configuration peaked at **71.8% Macro-F1** on the development set using: `Batch Size = 32`, `Max Length = 150`, `Suffix Length = 5`, and `Epochs = 20`.

---

## Final Test Set Results & Analysis

Following strict evaluation protocols, the champion model (`NN_bs32_ml150_sl5_ep20`) was evaluated exactly once on the unseen test set to measure true generalization.

**Overall Test Scores:**
* **Macro-Average F1:** 69.8%
* **Micro-Average F1:** 88.0%

**Per-Class Breakdown:**
* **brand:** 89.0% F1
* **drug:** 92.8% F1
* **group:** 81.0% F1
* **drug_n:** 16.3% F1

**Discussion & Key Takeaways:**
1. **Generalization Success:** The model only dropped 2.0% between the development set (71.8%) and the unseen test set (69.8%). This proves the LayerNorm and Dropout additions successfully prevented the GloVe embeddings from simply memorizing the training data.
2. **Exceptional Core Accuracy:** The model is highly reliable on standard medical entities, identifying approved `drug` mentions with nearly 93% accuracy and hitting an 88.0% overall Micro-F1. The GloVe embeddings specifically helped the model understand the abstract contexts surrounding `group` entities (81.0%).
3. **The `drug_n` Bottleneck:** The primary reason the Macro-average is not in the 80s is due to the `drug_n` class (unapproved/experimental substances). With only 102 examples in the entire test set, the data-hungry neural network failed to learn its highly irregular patterns. Because Macro-averaging weighs all four classes equally, this single data-starved class artificially dragged down the overall score.



in nn_glove_results_v2 there are all the final stats