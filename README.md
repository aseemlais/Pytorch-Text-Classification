# Medical Abstract Classification using BERT, PubMedBERT & LoRA

A Transformer-based Natural Language Processing project for **multi-class medical abstract classification** using pretrained BERT-family models and **parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)**.

The project follows a **systematic, hypothesis-driven experimental approach**. Instead of selecting a model and tuning parameters randomly, each experiment was designed based on the limitations observed in the previous experiment.

The study progresses from a general-domain BERT baseline to biomedical-domain pretraining and finally to parameter-efficient fine-tuning.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Class Distribution](#class-distribution)
- [Data Preprocessing](#data-preprocessing)
- [Experimental Methodology](#experimental-methodology)
  - [Experiment 1](#experiment-1--bert-base-with-weighted-crossentropy)
  - [Experiment 2](#experiment-2--bert-base-with-unweighted-crossentropy)
  - [Experiment 3](#experiment-3--bert-base-with-scheduling-and-gradient-clipping)
  - [Experiment 4](#experiment-4--pubmedbert-with-scheduling)
  - [Experiment 5](#experiment-5--pubmedbert-with-lora)
- [Experimental Comparison](#experimental-comparison)
- [Final Model](#final-model)
- [Final Validation Performance](#final-validation-performance)
- [Key Findings](#key-findings)
- [Training Configuration](#training-configuration)
- [Evaluation Strategy](#evaluation-strategy)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

# Overview

Medical text classification is challenging because medical abstracts contain highly specialized terminology, domain-specific semantic relationships, and imbalanced class distributions.

This project investigates the effectiveness of different Transformer-based approaches for classifying medical abstracts into five medical condition categories.

The experiments investigate four major aspects:

1. **Loss-function design** for class imbalance
2. **Optimization strategies** for model convergence and overfitting
3. **Biomedical domain pretraining** for improved text representations
4. **Parameter-efficient fine-tuning** using LoRA

The experimental progression was:


                    BERT-base
                       │
             ┌─────────┴─────────┐
             │                   │
       Weighted CE        Unweighted CE
             │                   │
             └─────────┬─────────┘
                       │
              Scheduler + Clipping
                       │
                       ▼
                  PubMedBERT
                       │
                       ▼
               PubMedBERT + LoRA
                       │
                       ▼
              Best Configuration


Objectives:
The main objectives of the project were:

Develop a Transformer-based medical text classification pipeline.
Establish a strong BERT baseline.
Investigate the effect of class-weighted loss.
Study the effect of learning-rate scheduling and gradient clipping.
Determine whether biomedical-domain pretraining improves performance.
Investigate whether LoRA can reduce overfitting during fine-tuning.
Evaluate performance using class-aware metrics such as Macro-F1.
Perform experiments in a controlled and reproducible manner.

Problem Statement

Given a medical abstract, predict its corresponding medical condition category.

The task contains five classes:

Neoplasms
Digestive system diseases
Nervous system diseases
Cardiovascular diseases
General pathological conditions

This is a supervised 5-class text classification problem.

The dataset is imbalanced, with General pathological conditions being the largest class.



Dataset: Medical Abstracts

Source: Hugging Face — TimSchopf/medical_abstracts

Dataset Statistics
Property	Value
Total samples	14,438
Original training samples	11,550
Held-out test samples	2,888
Number of classes	5

The original training set was split into training and validation subsets using stratified sampling.

Final Dataset Split
Split	Samples
Training	9,817
Validation	1,733
Test	2,888
Total	14,438
Split Configuration
Train / Validation split : 85 / 15
Stratification           : Yes
Random State              : 42

The original test set was kept completely untouched during model development.

Class Distribution

The five target classes are:

Label	Class
0	Neoplasms
1	Digestive system diseases
2	Nervous system diseases
3	Cardiovascular diseases
4	General pathological conditions

The dataset is imbalanced.

General pathological conditions is the largest class and was consistently one of the most difficult categories for the models to classify correctly.

Data Preprocessing

The same fundamental preprocessing strategy was maintained across the experiments to ensure fair comparison.

Tokenization

Each abstract was tokenized using the tokenizer associated with the selected pretrained Transformer model.

Maximum sequence length:

512 tokens

Sequences longer than 512 tokens were truncated.

Shorter sequences were padded.

Each sample contains:

input_ids
attention_mask
labels

The original labels were converted from 1–5 to zero-based indices 0–4 for compatibility with PyTorch CrossEntropyLoss.

Experimental Methodology

The project contains five main experiments.

Each experiment was designed to answer a specific question based on the results of the previous experiment.

Experiment 1 — BERT-base with Weighted CrossEntropy
Objective

Establish a baseline using a general-domain pretrained BERT model while explicitly addressing class imbalance through weighted CrossEntropyLoss.

Model
bert-base-uncased
Configuration
Loss          : Weighted CrossEntropyLoss
Optimizer     : AdamW
Learning Rate : 2e-5
Weight Decay  : 0.01
Class Weights

The class weights were calculated from the training subset:

[0.9132, 1.9325, 1.4999, 0.9462, 0.6010]
Result
Metric	Result
Validation Accuracy	61.74%
Validation Macro-F1	0.6189
Observation

Weighted loss increased the importance of minority classes but did not improve overall Macro-F1.

The largest class, General pathological conditions, was particularly affected.

Recall : 28.77%
F1     : 0.4000

The model also showed a significant train-validation gap, indicating early overfitting.

Conclusion

Class weighting redistributed errors between classes but did not address the underlying representation limitation.

Experiment 2 — BERT-base with Unweighted CrossEntropy
Objective

Determine whether class weighting itself was responsible for the poor performance.

The class weights were removed while keeping the remaining optimization configuration unchanged.

Configuration
Model         : bert-base-uncased
Loss          : CrossEntropyLoss
Optimizer     : AdamW
Learning Rate : 2e-5
Weight Decay  : 0.01
Result
Metric	Result
Validation Accuracy	63.19%
Validation Macro-F1	0.6268
Observation

Removing class weighting improved Macro-F1:

0.6189 → 0.6268

However, the model continued to overfit early during training.

Conclusion

Class weighting was not the primary solution to the problem.

The next experiment therefore focused on optimization behavior.

Experiment 3 — BERT-base with Scheduling and Gradient Clipping
Objective

Investigate whether improved optimization could reduce overfitting and improve generalization.

Additional Techniques
Unweighted CrossEntropyLoss
AdamW
10% linear learning-rate warm-up
Linear learning-rate decay
Gradient clipping
Maximum gradient norm: 1.0
Early stopping
Result
Metric	Result
Validation Accuracy	63.94%
Validation Macro-F1	0.6347
Observation

The optimization improvements provided a modest improvement:

0.6268 → 0.6347 Macro-F1

However, the overfitting pattern remained.

Training performance continued improving while validation performance degraded after the early epochs.

Conclusion

Optimization strategies improved convergence but did not address the primary limitation.

The working hypothesis shifted toward domain representation mismatch.

General-domain BERT was being fine-tuned on a relatively small dataset containing highly specialized biomedical language.

Experiment 4 — PubMedBERT with Scheduling
Objective

Directly test whether biomedical-domain pretraining provides better representations for medical text classification.

Model
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
Configuration
Loss              : Unweighted CrossEntropyLoss
Optimizer         : AdamW
Learning Rate     : 2e-5
Weight Decay      : 0.01
Warm-up            : 10%
Scheduler          : Linear decay
Gradient Clipping : 1.0
Early Stopping    : Yes
Fine-tuning       : Full model
Result
Metric	Result
Validation Accuracy	66.24%
Validation Macro-F1	0.6509
Improvement

Compared with the previous BERT-base configuration:

0.6347 → 0.6509 Macro-F1
Observation

The improvement supported the domain-representation hypothesis.

Biomedical-domain pretraining provided better representations for this medical classification task than general-domain BERT.

However, full fine-tuning still exhibited significant overfitting.

This motivated the next experiment.

Experiment 5 — PubMedBERT with LoRA
Objective

Address the overfitting observed during full fine-tuning using parameter-efficient fine-tuning.

LoRA

LoRA stands for:

Low-Rank Adaptation

Instead of updating all pretrained PubMedBERT parameters, the pretrained encoder was frozen and low-rank trainable adapters were introduced into selected attention projections.

This significantly reduced the number of parameters being updated.

LoRA Configuration
Rank (r)       : 8
Alpha          : 16
Dropout        : 0.1
Target Modules : Query + Value projections
Learning Rates

Different learning rates were used for the LoRA adapters and classification head.

Parameter Group	Learning Rate
LoRA parameters	2e-4
Classification head	2e-5
Training Configuration
Base Model            : PubMedBERT
Fine-Tuning Method    : LoRA
Loss                  : CrossEntropyLoss
Optimizer             : AdamW
Weight Decay          : 0.01
Warm-up               : 10%
Scheduler             : Linear decay
Gradient Clipping     : 1.0
Batch Size            : 16
Maximum Epochs        : 12
Early Stopping        : Patience 3
Padding               : Dynamic
Mixed Precision       : FP16
Selection Metric      : Validation Macro-F1

The best checkpoint occurred at Epoch 3 during the six-epoch training run.

Result
Metric	Result
Validation Accuracy	66.65%
Validation Macro-F1	0.6630
Validation Weighted-F1	0.6525
Experimental Comparison
Experiment	Model	Main Technique	Val Accuracy	Val Macro-F1
1	BERT-base	Weighted CE	61.74%	0.6189
2	BERT-base	Unweighted CE	63.19%	0.6268
3	BERT-base	Scheduler + Clipping	63.94%	0.6347
4	PubMedBERT	Scheduler + Clipping	66.24%	0.6509
5	PubMedBERT + LoRA	Parameter-efficient fine-tuning	66.65%	0.6630
Overall Improvement

Initial baseline:

Macro-F1 = 0.6189

Best configuration:

Macro-F1 = 0.6630

Absolute improvement:

0.6630 - 0.6189 = 0.0441

Therefore:

Overall improvement = +4.41 percentage points in Macro-F1

Final Validation Performance

The best PubMedBERT + LoRA configuration achieved:

Metric	Score
Accuracy	66.65%
Macro-F1	66.30%
Weighted-F1	65.25%
Per-Class Performance
Class	Precision	Recall	F1
Neoplasms	0.7030	0.9158	0.7954
Digestive system diseases	0.5462	0.7933	0.6469
Nervous system diseases	0.5830	0.6234	0.6025
Cardiovascular diseases	0.7389	0.7732	0.7557
General pathological conditions	0.6839	0.4125	0.5146
Key Findings
1. Weighted loss was not effective

Weighted CrossEntropyLoss did not improve overall Macro-F1.

Although minority-class recall improved, the largest class experienced poor recall.

This demonstrated that class weighting alone could not solve the underlying classification difficulty.

2. Unweighted loss performed better

Removing class weighting improved Macro-F1:

0.6189 → 0.6268

This showed that weighting was not beneficial for this particular dataset and model configuration.

3. Optimization improvements produced modest gains

Learning-rate warm-up, linear decay, gradient clipping, and early stopping improved the BERT-base result:

0.6268 → 0.6347

However, these techniques did not eliminate early overfitting.

4. Biomedical pretraining produced a significant improvement

Replacing general-domain BERT with PubMedBERT increased Macro-F1:

0.6347 → 0.6509

This supported the hypothesis that domain-specific pretrained representations are more suitable for medical text classification.

5. LoRA provided an additional improvement

Applying LoRA to PubMedBERT increased Macro-F1:

0.6509 → 0.6630

The improvement was achieved while updating only a small fraction of the pretrained model parameters.

6. LoRA helped control the overfitting pattern

Full fine-tuning experiments showed a rapid increase in training performance while validation performance degraded.

With LoRA, the train-validation gap remained smaller for longer.

Around the best region of training:

Train Loss      ≈ 0.822
Validation Loss ≈ 0.830

This suggests that restricting the number of trainable parameters acted as a structural regularizer.

7. General pathological conditions remains the primary challenge

General pathological conditions remained the weakest category.

However, its performance improved substantially compared with the initial experiment.

F1 Improvement
0.4000 → 0.5146
Recall Improvement
28.77% → 41.25%

Despite the improvement, the class remains significantly harder to classify than the other categories.

One possible explanation is that the class is broad and heterogeneous, causing semantic overlap with several other medical categories.

Final Model Architecture
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
      Frozen Parameters       LoRA Adapters
                                   │
                              Query + Value
                                   │
                                   ▼
                          [CLS] Representation
                                   │
                                   ▼
                         Classification Head
                                   │
                                   ▼
                            5 Class Logits
Training Configuration
Data
Dataset              : TimSchopf/medical_abstracts
Training Samples     : 9,817
Validation Samples   : 1,733
Test Samples         : 2,888
Classes              : 5
Max Sequence Length  : 512
Random State         : 42
Final Model
Base Model           : PubMedBERT
Fine-Tuning          : LoRA
LoRA Rank            : 8
LoRA Alpha           : 16
LoRA Dropout         : 0.1
Target Modules       : Query + Value
Optimization
Optimizer             : AdamW
LoRA Learning Rate    : 2e-4
Classifier LR         : 2e-5
Weight Decay          : 0.01
Warm-up               : 10%
Scheduler              : Linear decay
Gradient Clipping     : 1.0
Early Stopping        : Patience 3
Efficiency
Batch Size             : 16
Padding                : Dynamic
Mixed Precision        : FP16
Hardware               : NVIDIA T4
Platform               : Google Colab
Evaluation Strategy

Validation Macro-F1 was used as the primary model-selection metric.

Macro-F1 was chosen because the dataset is imbalanced and gives equal importance to each class.

Accuracy and Weighted-F1 were also tracked as secondary metrics.

The evaluation process was:

Original Training Data
          │
          ▼
 Stratified 85/15 Split
          │
     ┌────┴────┐
     │         │
 Training   Validation
     │         │
     │         ▼
     │   Model Selection
     │   using Macro-F1
     │         │
     └─────────┘
               │
               ▼
       Final Configuration
               │
               ▼
       Held-out Test Set

The test set was kept untouched throughout model development and reserved for final evaluation.

Training Environment
Platform          : Google Colab Free Tier
GPU               : NVIDIA T4
Mixed Precision   : FP16
Transformers      : 5.17.0

The experiments were performed using a single NVIDIA T4 GPU.

Transformers v5 Compatibility

The project was developed using Transformers 5.17.0.

Several API changes introduced in Transformers v5 required updated arguments.

Examples include:

warmup_ratio
      ↓
warmup_steps

evaluation_strategy
      ↓
eval_strategy

Trainer(tokenizer=...)
      ↓
Trainer(processing_class=...)
Technologies Used
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT / LoRA
Scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
Google Colab
NVIDIA T4 GPU
 
Large datasets, model checkpoints, cache files, and generated artifacts should be excluded from Git using .gitignore.

Reproducibility

To reproduce the experiments:

Install the dependencies listed in requirements.txt.
Load the TimSchopf/medical_abstracts dataset.
Separate the original training and test sets.
Create the stratified 85/15 train-validation split.
Use random_state=42.
Use the tokenizer corresponding to the selected pretrained model.
Apply the configuration associated with the desired experiment.
Train using validation Macro-F1 as the model-selection metric.
Save the best-performing checkpoint.
Evaluate the final selected model on the untouched test set.

For fair comparison, the dataset split and evaluation protocol should remain unchanged across experiments.

Limitations
The dataset is imbalanced.
General pathological conditions remains substantially harder to classify.
Experiments were primarily conducted using a single random seed.
Multi-seed statistical evaluation was not performed.
Hyperparameter sweeps were limited due to computational constraints.
Training was performed using a single NVIDIA T4 GPU.
Model performance depends on the quality and consistency of the dataset labels.
Validation performance may not fully represent generalization to unseen medical text.
Future Work

Potential future improvements include:

BioBERT comparison
Discriminative learning rates
Layer-wise learning-rate decay
Multi-seed evaluation
Hyperparameter optimization
Detailed error analysis
Dataset and label-quality auditing
Analysis of General pathological conditions
Biomedical instruction-tuned models
Alternative parameter-efficient fine-tuning methods
Model explainability
Confidence calibration
API deployment
Interactive inference application
Conclusion

This project demonstrates a systematic and diagnosis-driven approach to Transformer-based medical text classification.

Instead of simply increasing model size or changing hyperparameters randomly, each experiment was motivated by the limitations observed in the previous configuration:

Weighted CrossEntropy
        ↓
Unweighted CrossEntropy
        ↓
Learning-rate Scheduling
        ↓
Biomedical Domain Pretraining
        ↓
Parameter-Efficient Fine-Tuning
        ↓
PubMedBERT + LoRA

👤 Author

Aseem Lais T P
