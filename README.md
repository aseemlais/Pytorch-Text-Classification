# Medical Abstract Classification using BERT, PubMedBERT & LoRA

A Transformer-based NLP project for multi-class medical abstract classification, progressing from a general-domain BERT baseline to biomedical-domain pretraining and finally to parameter-efficient fine-tuning with **LoRA (Low-Rank Adaptation)**.

The project follows a systematic, **diagnosis-driven experimental approach**: each experiment was designed to test a specific hypothesis raised by the limitations observed in the previous one, rather than tuning hyperparameters at random.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Experimental Methodology](#experimental-methodology)
  - [Experiment 1 — Weighted CrossEntropy](#experiment-1--bert-base-with-weighted-crossentropy)
  - [Experiment 2 — Unweighted CrossEntropy](#experiment-2--bert-base-with-unweighted-crossentropy)
  - [Experiment 3 — Scheduling & Gradient Clipping](#experiment-3--bert-base-with-scheduling--gradient-clipping)
  - [Experiment 4 — PubMedBERT](#experiment-4--pubmedbert-with-scheduling)
  - [Experiment 5 — PubMedBERT + LoRA](#experiment-5--pubmedbert-with-lora)
- [Experimental Comparison](#experimental-comparison)
- [Final Model](#final-model)
- [Final Validation Performance](#final-validation-performance)
- [Key Findings](#key-findings)
- [Training Configuration](#training-configuration)
- [Evaluation Strategy](#evaluation-strategy)
- [Training Environment](#training-environment)
- [Technologies Used](#technologies-used)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Author](#author)

---

## Overview

Medical text classification is challenging: abstracts contain specialized terminology, domain-specific semantic relationships, and imbalanced class distributions. This project investigates the effectiveness of different Transformer-based approaches for classifying medical abstracts into five medical condition categories.

The experiments investigate four major aspects:

1. Loss-function design for class imbalance
2. Optimization strategies for convergence and overfitting
3. Biomedical-domain pretraining for improved text representations
4. Parameter-efficient fine-tuning using LoRA

**Experimental progression:**

```
                    BERT-base
                        │
          ┌─────────────┴─────────────┐
          │                           │
    Weighted CE                Unweighted CE
          │                           │
          └─────────────┬─────────────┘
                        │
             Scheduler + Gradient Clipping
                        │
                        ▼
                   PubMedBERT
                        │
                        ▼
               PubMedBERT + LoRA
                        │
                        ▼
               Best Configuration
```

---

## Objectives

- Develop a Transformer-based medical text classification pipeline
- Establish a strong BERT baseline
- Investigate the effect of class-weighted loss
- Study the effect of learning-rate scheduling and gradient clipping
- Determine whether biomedical-domain pretraining improves performance
- Investigate whether LoRA can reduce overfitting during fine-tuning
- Evaluate performance using class-aware metrics such as Macro-F1
- Perform all experiments in a controlled, reproducible manner

---

## Problem Statement

Given a medical abstract, predict its corresponding medical condition category — a supervised 5-class text classification problem.

| Label | Class |
|---|---|
| 0 | Neoplasms |
| 1 | Digestive system diseases |
| 2 | Nervous system diseases |
| 3 | Cardiovascular diseases |
| 4 | General pathological conditions |

The dataset is imbalanced, with **General pathological conditions** being the largest class — and, as the experiments show, consistently one of the hardest to classify correctly.

---

## Dataset

**Source:** Hugging Face — [`TimSchopf/medical_abstracts`](https://huggingface.co/datasets/TimSchopf/medical_abstracts)

| Property | Value |
|---|---|
| Total samples | 14,438 |
| Original training samples | 11,550 |
| Held-out test samples | 2,888 |
| Number of classes | 5 |

The original training set was split into training and validation subsets using **stratified sampling**.

| Split | Samples |
|---|---|
| Training | 9,817 |
| Validation | 1,733 |
| Test | 2,888 |
| **Total** | **14,438** |

**Split configuration:**
- Train / Validation split: 85 / 15
- Stratification: Yes
- Random state: 42

The original test set was kept **completely untouched** during model development and used only for final evaluation.

---

## Data Preprocessing

The same preprocessing strategy was maintained across all experiments to ensure a fair comparison.

- **Tokenization:** each abstract tokenized using the tokenizer of the selected pretrained model
- **Maximum sequence length:** 512 tokens
- Sequences longer than 512 tokens were truncated; shorter sequences were padded
- Each sample contains: `input_ids`, `attention_mask`, `labels`
- Original labels (1–5) were converted to zero-indexed labels (0–4) for compatibility with PyTorch `CrossEntropyLoss`

---

## Experimental Methodology

Five experiments were run, each designed to answer a specific question raised by the previous result.

### Experiment 1 — BERT-base with Weighted CrossEntropy

**Objective:** Establish a baseline using a general-domain pretrained BERT model while explicitly addressing class imbalance via weighted CrossEntropyLoss.

**Configuration**

| Setting | Value |
|---|---|
| Model | `bert-base-uncased` |
| Loss | Weighted CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Class Weights | `[0.9132, 1.9325, 1.4999, 0.9462, 0.6010]` |

**Result**

| Metric | Result |
|---|---|
| Validation Accuracy | 61.74% |
| Validation Macro-F1 | 0.6189 |

**Observation:** Weighted loss increased the importance of minority classes but did not improve overall Macro-F1. The largest class, *General pathological conditions*, was particularly affected — Recall: 28.77%, F1: 0.4000. The model also showed a significant train-validation gap, indicating early overfitting.

**Conclusion:** Class weighting redistributed errors between classes but did not address the underlying representation limitation.

---

### Experiment 2 — BERT-base with Unweighted CrossEntropy

**Objective:** Determine whether class weighting itself was responsible for the poor performance. Class weights were removed; all other settings held constant.

**Configuration**

| Setting | Value |
|---|---|
| Model | `bert-base-uncased` |
| Loss | CrossEntropyLoss (unweighted) |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |

**Result**

| Metric | Result |
|---|---|
| Validation Accuracy | 63.19% |
| Validation Macro-F1 | 0.6268 |

**Observation:** Removing class weighting improved Macro-F1 (0.6189 → 0.6268), but the model continued to overfit early during training.

**Conclusion:** Class weighting was not the primary solution. The next experiment focused on optimization behavior.

---

### Experiment 3 — BERT-base with Scheduling & Gradient Clipping

**Objective:** Investigate whether improved optimization could reduce overfitting and improve generalization.

**Additional Techniques**
- Unweighted CrossEntropyLoss
- AdamW
- 10% linear learning-rate warm-up
- Linear learning-rate decay
- Gradient clipping (max norm: 1.0)
- Early stopping

**Result**

| Metric | Result |
|---|---|
| Validation Accuracy | 63.94% |
| Validation Macro-F1 | 0.6347 |

**Observation:** Optimization improvements provided a modest gain (0.6268 → 0.6347), but the overfitting pattern remained — training performance kept improving while validation performance degraded after the early epochs.

**Conclusion:** Optimization strategies improved convergence but did not address the primary limitation. The working hypothesis shifted toward **domain representation mismatch** — general-domain BERT was being fine-tuned on a relatively small dataset of highly specialized biomedical language.

---

### Experiment 4 — PubMedBERT with Scheduling

**Objective:** Directly test whether biomedical-domain pretraining provides better representations for medical text classification.

**Configuration**

| Setting | Value |
|---|---|
| Model | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Loss | Unweighted CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warm-up | 10% |
| Scheduler | Linear decay |
| Gradient Clipping | 1.0 |
| Early Stopping | Yes |
| Fine-tuning | Full model |

**Result**

| Metric | Result |
|---|---|
| Validation Accuracy | 66.24% |
| Validation Macro-F1 | 0.6509 |

**Observation:** A significant improvement over BERT-base (0.6347 → 0.6509), supporting the domain-representation hypothesis. However, full fine-tuning still exhibited significant overfitting — motivating the next experiment.

---

### Experiment 5 — PubMedBERT with LoRA

**Objective:** Address the overfitting observed during full fine-tuning using parameter-efficient fine-tuning.

LoRA (**Low-Rank Adaptation**) freezes the pretrained encoder and introduces small trainable low-rank adapters into selected attention projections, drastically reducing the number of parameters updated during training.

**LoRA Configuration**

| Setting | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Target Modules | Query + Value projections |

**Learning Rates**

| Parameter Group | Learning Rate |
|---|---|
| LoRA parameters | 2e-4 |
| Classification head | 2e-5 |

**Training Configuration**

| Setting | Value |
|---|---|
| Base Model | PubMedBERT |
| Fine-Tuning Method | LoRA |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warm-up | 10% |
| Scheduler | Linear decay |
| Gradient Clipping | 1.0 |
| Batch Size | 16 |
| Maximum Epochs | 12 |
| Early Stopping | Patience 3 |
| Padding | Dynamic |
| Mixed Precision | FP16 |
| Selection Metric | Validation Macro-F1 |

The best checkpoint occurred at **Epoch 3** during a six-epoch training run.

**Result**

| Metric | Result |
|---|---|
| Validation Accuracy | 66.65% |
| Validation Macro-F1 | 0.6630 |
| Validation Weighted-F1 | 0.6525 |

---

## Experimental Comparison

| Experiment | Model | Main Technique | Val Accuracy | Val Macro-F1 |
|---|---|---|---|---|
| 1 | BERT-base | Weighted CE | 61.74% | 0.6189 |
| 2 | BERT-base | Unweighted CE | 63.19% | 0.6268 |
| 3 | BERT-base | Scheduler + Clipping | 63.94% | 0.6347 |
| 4 | PubMedBERT | Scheduler + Clipping | 66.24% | 0.6509 |
| **5** | **PubMedBERT + LoRA** | **Parameter-efficient fine-tuning** | **66.65%** | **0.6630** ⭐ |

**Overall improvement:**

$$0.6630 - 0.6189 = 0.0441$$

→ **+4.41 percentage points** in Macro-F1 from the initial baseline to the final configuration.

---

## Final Model

**Architecture**

```
             Medical Abstract
                    │
                    ▼
          PubMedBERT Tokenizer
                    │
                    ▼
           PubMedBERT Encoder
                    │
         ┌──────────┴──────────┐
         │                     │
   Frozen Parameters      LoRA Adapters
                           (Query + Value)
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
          [CLS] Representation
                    │
                    ▼
          Classification Head
                    │
                    ▼
             5 Class Logits
```

---

## Final Validation Performance

**Overall**

| Metric | Score |
|---|---|
| Accuracy | 66.65% |
| Macro-F1 | 66.30% |
| Weighted-F1 | 65.25% |

**Per-Class Performance**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Neoplasms | 0.7030 | 0.9158 | 0.7954 |
| Digestive system diseases | 0.5462 | 0.7933 | 0.6469 |
| Nervous system diseases | 0.5830 | 0.6234 | 0.6025 |
| Cardiovascular diseases | 0.7389 | 0.7732 | 0.7557 |
| General pathological conditions | 0.6839 | 0.4125 | 0.5146 |

---

## Key Findings

**1. Weighted loss was not effective**
Weighted CrossEntropyLoss did not improve overall Macro-F1. Although minority-class recall improved, the largest class experienced poor recall — demonstrating that class weighting alone could not solve the underlying classification difficulty.

**2. Unweighted loss performed better**
Removing class weighting improved Macro-F1 from 0.6189 to 0.6268, showing that weighting was not beneficial for this dataset/model configuration.

**3. Optimization improvements produced modest gains**
Learning-rate warm-up, linear decay, gradient clipping, and early stopping improved the BERT-base result from 0.6268 to 0.6347 — but did not eliminate early overfitting.

**4. Biomedical pretraining produced a significant improvement**
Replacing general-domain BERT with PubMedBERT increased Macro-F1 from 0.6347 to 0.6509, supporting the hypothesis that domain-specific pretrained representations are better suited to medical text classification.

**5. LoRA provided an additional improvement**
Applying LoRA to PubMedBERT increased Macro-F1 from 0.6509 to 0.6630 — while updating only a small fraction of the pretrained model's parameters.

**6. LoRA helped control the overfitting pattern**
Full fine-tuning experiments showed training performance rising rapidly while validation performance degraded. With LoRA, the train-validation gap remained smaller for longer (around the best region: Train Loss ≈ 0.822, Validation Loss ≈ 0.830) — suggesting that restricting trainable parameters acted as a structural regularizer.

**7. "General pathological conditions" remains the primary challenge**
This class remained the weakest category throughout, though it improved substantially over the course of the project:

| Metric | Before (Exp 1) | After (Exp 5) |
|---|---|---|
| F1 | 0.4000 | 0.5146 |
| Recall | 28.77% | 41.25% |

One likely explanation: the class is broad and heterogeneous, causing semantic overlap with several other medical categories.

---

## Training Configuration

**Data**

| Setting | Value |
|---|---|
| Dataset | `TimSchopf/medical_abstracts` |
| Training Samples | 9,817 |
| Validation Samples | 1,733 |
| Test Samples | 2,888 |
| Classes | 5 |
| Max Sequence Length | 512 |
| Random State | 42 |

**Final Model**

| Setting | Value |
|---|---|
| Base Model | PubMedBERT |
| Fine-Tuning | LoRA |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Target Modules | Query + Value |

**Optimization**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LoRA Learning Rate | 2e-4 |
| Classifier Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warm-up | 10% |
| Scheduler | Linear decay |
| Gradient Clipping | 1.0 |
| Early Stopping | Patience 3 |

**Efficiency**

| Setting | Value |
|---|---|
| Batch Size | 16 |
| Padding | Dynamic |
| Mixed Precision | FP16 |
| Hardware | NVIDIA T4 |
| Platform | Google Colab |

---

## Evaluation Strategy

Validation Macro-F1 was used as the primary model-selection metric, chosen because the dataset is imbalanced and Macro-F1 gives equal weight to each class. Accuracy and Weighted-F1 were tracked as secondary metrics.

```
       Original Training Data
                │
                ▼
      Stratified 85/15 Split
                │
       ┌────────┴────────┐
       │                 │
   Training          Validation
       │                 │
       ▼                 │
     Model Selection      │
     (via Macro-F1) ◄─────┘
       │
       ▼
  Final Configuration
       │
       ▼
   Held-out Test Set
```

The test set was kept untouched throughout model development and reserved solely for final evaluation.

---

## Training Environment

| Setting | Value |
|---|---|
| Platform | Google Colab (Free Tier) |
| GPU | NVIDIA T4 |
| Mixed Precision | FP16 |
| Transformers | 5.17.0 |

**Transformers v5 compatibility notes**

Several API changes introduced in Transformers v5 required updated arguments:

| v4-style | v5 replacement |
|---|---|
| `warmup_ratio` | `warmup_steps` (computed as an integer) |
| `evaluation_strategy` | `eval_strategy` |
| `Trainer(tokenizer=...)` | `Trainer(processing_class=...)` |

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Google Colab (NVIDIA T4 GPU)

> Large datasets, model checkpoints, cache files, and generated artifacts should be excluded from Git via `.gitignore`.

---

## Reproducibility

To reproduce the experiments:

1. Install the dependencies listed in `requirements.txt`.
2. Load the `TimSchopf/medical_abstracts` dataset.
3. Separate the original training and test sets.
4. Create the stratified 85/15 train-validation split using `random_state=42`.
5. Use the tokenizer corresponding to the selected pretrained model.
6. Apply the configuration associated with the desired experiment.
7. Train using validation Macro-F1 as the model-selection metric.
8. Save the best-performing checkpoint.
9. Evaluate the final selected model on the untouched test set.

For fair comparison, the dataset split and evaluation protocol should remain unchanged across all experiments.

---

## Limitations

- The dataset is imbalanced.
- *General pathological conditions* remains substantially harder to classify than other categories.
- Experiments were primarily conducted using a single random seed; multi-seed statistical evaluation was not performed.
- Hyperparameter sweeps were limited due to computational constraints.
- Training was performed using a single NVIDIA T4 GPU.
- Model performance depends on the quality and consistency of the dataset labels.
- Validation performance may not fully represent generalization to unseen medical text.

---

## Future Work

- BioBERT comparison
- Discriminative learning rates
- Layer-wise learning-rate decay
- Multi-seed evaluation
- Hyperparameter optimization
- Detailed error analysis
- Dataset and label-quality auditing
- Deeper analysis of *General pathological conditions*
- Biomedical instruction-tuned models
- Alternative parameter-efficient fine-tuning methods
- Model explainability
- Confidence calibration
- API deployment
- Interactive inference application

---

## Conclusion

This project demonstrates a systematic, diagnosis-driven approach to Transformer-based medical text classification. Rather than increasing model size or changing hyperparameters at random, each experiment was motivated directly by the limitations observed in the previous configuration:

```
Weighted CrossEntropy
        │
        ▼
Unweighted CrossEntropy
        │
        ▼
Learning-rate Scheduling
        │
        ▼
Biomedical Domain Pretraining
        │
        ▼
Parameter-Efficient Fine-Tuning
        │
        ▼
   PubMedBERT + LoRA
```

---

## Author

**Aseem Lais T P**
