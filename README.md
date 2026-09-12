AuraMetrics

A custom multimodal fusion model for sentiment analysis, built as part of a Capstone project at Uni. AuraMetrics fuses text, audio, and visual signals to predict sentiment, with the goal of applying multimodal analysis to retail customer feedback.

Overview

Most sentiment analysis systems rely on text alone. AuraMetrics combines three modalities — spoken/written language, vocal tone, and facial expression — into a single model, on the idea that customer sentiment is often expressed more clearly across channels than in text alone.

The model is trained and evaluated on CMU-MOSEI, a standard benchmark dataset for multimodal sentiment analysis.

Architecture
Text: DistilBERT embeddings
Audio: COVAREP acoustic features, encoded via a bidirectional LSTM
Visual: OpenFace 2.0 facial action unit features, encoded via a bidirectional LSTM
Fusion: Cross-modal attention across all modality pairs (text↔audio, text↔visual), inspired by the MISA architecture
Robustness: Modality dropout during training, so the model degrades gracefully rather than failing when a modality is missing or noisy at inference — a common real-world constraint that most fusion models ignore
Output: Continuous sentiment regression head
Results

Evaluated on a held-out CMU-MOSEI test split:

Metric	Score
* MAE	0.4748
* Pearson Correlation	0.6093
* 7-class Accuracy (Acc-7)	53.40%
* Binary Accuracy, pos/neg (Acc-2)	81.51%
* F1, pos/neg	81.21%
* Binary Accuracy, non-neg/neg (Acc-2)	81.80%
* Training time	64.5 minutes

Project Structure \


capstone_model \
├── model.py &emsp; &emsp; &emsp; &emsp; &emsp; # AuraMetricsFusionModel architecture \
├── dataset.py &emsp; &emsp; &emsp; &emsp; &nbsp; # CMU-MOSEI dataset loading and preprocessing \
├── train.py &emsp; &emsp; &emsp; &emsp; &ensp; &emsp;# Training loop, early stopping, LR scheduling \
├── DistilBERT.ipynb &emsp; &emsp; # Text feature extraction notebook \
├── results.txt &emsp; &emsp; &emsp; &emsp; &nbsp;  # Final evaluation metrics \
└── best_model.pt &emsp; &emsp; &ensp; # Saved model checkpoint 

Training Setup
* Optimizer: AdamW
* Scheduler: ReduceLROnPlateau
* Early stopping (patience: 7 epochs)
* Train/dev/test split: ~2,000 / 300 / 500 samples (CMU-MOSEI subset)

