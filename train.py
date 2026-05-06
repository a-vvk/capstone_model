"""
AuraMetrics Custom Multimodal Fusion Model — Training Script
============================================================
Trains a lightweight cross-modal attention fusion model on CMU-MOSEI
using pre-extracted DistilBERT text features, COVAREP audio features,
and OpenFace 2.0 visual features.

Usage:
    python train.py

Results are printed to console and saved to results.txt.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import get_loaders
from model import AuraMetricsFusionModel


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    'data_dir':        '/Users/jl/MISA/datasets/MOSEI',
    'batch_size':      16,

    # Model architecture
    'hidden_dim':      128,
    'audio_dim':       74,   # COVAREP features
    'visual_dim':      713,  # OpenFace 2.0 features
    'dropout':         0.3,
    'modality_dropout': 0.2, # probability of dropping a modality during training

    # Training
    'learning_rate':   1e-4,
    'weight_decay':    1e-4,
    'n_epochs':        30,
    'patience':        7,
    'clip':            1.0,

    # Output
    'checkpoint_path': 'best_model.pt',
    'results_path':    'results.txt',
}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, to_print=False):
    """
    Compute standard multimodal sentiment analysis metrics.
    Matches the evaluation protocol used in MISA and related work.
    """
    mae  = np.mean(np.abs(y_pred - y_true))
    corr = np.corrcoef(y_pred, y_true)[0][1]

    # Binary accuracy (positive vs negative, excluding neutral)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
    if len(non_zeros) > 0:
        binary_truth = (y_true[non_zeros] > 0)
        binary_preds = (y_pred[non_zeros] > 0)
        acc_2    = accuracy_score(binary_truth, binary_preds)
        f1_2     = f1_score(binary_truth, binary_preds, average='weighted')
    else:
        acc_2, f1_2 = 0.0, 0.0

    # Non-negative vs negative accuracy
    binary_truth_nn = (y_true >= 0)
    binary_preds_nn = (y_pred >= 0)
    acc_nn = accuracy_score(binary_truth_nn, binary_preds_nn)

    # 7-class accuracy
    y_true_7 = np.clip(np.round(y_true), -3, 3)
    y_pred_7 = np.clip(np.round(y_pred), -3, 3)
    acc_7 = accuracy_score(y_true_7, y_pred_7)

    if to_print:
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"MAE:                    {mae:.4f}")
        print(f"Pearson Correlation:    {corr:.4f}")
        print(f"Acc-7 (7-class):        {acc_7:.4f}")
        print(f"Acc-2 (pos/neg):        {acc_2:.4f}")
        print(f"F1 (pos/neg):           {f1_2:.4f}")
        print(f"Acc-2 (non-neg/neg):    {acc_nn:.4f}")
        print(f"{'='*50}")
        if len(non_zeros) > 0:
            print("\nClassification Report (pos/neg):")
            print(classification_report(binary_truth, binary_preds, digits=5))

    return {
        'mae': mae, 'corr': corr,
        'acc_7': acc_7, 'acc_2': acc_2,
        'f1_2': f1_2, 'acc_nn': acc_nn
    }


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, clip):
    model.train()
    total_loss = 0
    n_batches  = len(loader)

    for batch_i, batch in enumerate(loader):
        optimizer.zero_grad()

        bert   = batch['bert_feat'].to(device)
        visual = batch['visual'].to(device)
        audio  = batch['acoustic'].to(device)
        v_lens = batch['visual_lens'].to(device)
        a_lens = batch['acoustic_lens'].to(device)
        labels = batch['label'].to(device)

        preds = model(bert, audio, visual, a_lens, v_lens)
        loss  = criterion(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

        if (batch_i + 1) % 5 == 0 or (batch_i + 1) == n_batches:
            print(f"  Batch {batch_i+1}/{n_batches} | Loss: {total_loss/(batch_i+1):.4f}", end='\r')

    print()
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            bert   = batch['bert_feat'].to(device)
            visual = batch['visual'].to(device)
            audio  = batch['acoustic'].to(device)
            v_lens = batch['visual_lens'].to(device)
            a_lens = batch['acoustic_lens'].to(device)
            labels = batch['label'].to(device)

            preds = model(bert, audio, visual, a_lens, v_lens)
            loss  = criterion(preds, labels)
            total_loss += loss.item()

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true).squeeze()
    y_pred = np.concatenate(y_pred).squeeze()

    return total_loss / len(loader), y_true, y_pred


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, dev_loader, test_loader = get_loaders(
        CONFIG['data_dir'], CONFIG['batch_size']
    )

    # Model
    print("\nBuilding model...")
    model = AuraMetricsFusionModel(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_dev_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    print(f"\n{'='*50}")
    print("TRAINING START")
    print(f"{'='*50}")

    for epoch in range(1, CONFIG['n_epochs'] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{CONFIG['n_epochs']}")
        print("-" * 30)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CONFIG['clip']
        )

        # Evaluate on dev set
        dev_loss, dev_true, dev_pred = evaluate(model, dev_loader, criterion, device)
        dev_metrics = compute_metrics(dev_true, dev_pred)

        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
              f"Dev Acc-2: {dev_metrics['acc_2']:.4f} | "
              f"Dev MAE: {dev_metrics['mae']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        scheduler.step(dev_loss)

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print(f"  ✓ New best model saved (dev loss: {dev_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG['patience']:
                print("Early stopping.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(CONFIG['checkpoint_path']))
    test_loss, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_true, test_pred, to_print=True)

    # Save results
    with open(CONFIG['results_path'], 'w') as f:
        f.write("AuraMetrics Custom Multimodal Fusion Model — Results\n")
        f.write("="*50 + "\n")
        f.write(f"MAE:                 {test_metrics['mae']:.4f}\n")
        f.write(f"Pearson Correlation: {test_metrics['corr']:.4f}\n")
        f.write(f"Acc-7:               {test_metrics['acc_7']:.4f}\n")
        f.write(f"Acc-2 (pos/neg):     {test_metrics['acc_2']:.4f}\n")
        f.write(f"F1 (pos/neg):        {test_metrics['f1_2']:.4f}\n")
        f.write(f"Acc-2 (non-neg/neg): {test_metrics['acc_nn']:.4f}\n")
        f.write(f"Training time:       {total_time/60:.1f} minutes\n")
    print(f"\nResults saved to {CONFIG['results_path']}")


if __name__ == '__main__':
    main()