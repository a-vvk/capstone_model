"""
AuraMetrics Custom Multimodal Fusion Model — Training Script (v4)
=================================================================
Multi-task model: predicts both sentiment (-3 to +3) and
6 emotion intensities (happiness, sadness, anger, fear, disgust, surprise).

Usage:
    python train.py
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import get_loaders, EMOTION_NAMES
from model import AuraMetricsFusionModel


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':         '/Users/jl/MISA/datasets/MOSEI',
    'batch_size':       16,
    'hidden_dim':       128,
    'audio_dim':        74,
    'visual_dim':       713,
    'dropout':          0.3,
    'modality_dropout': 0.2,
    'learning_rate':    1e-4,
    'weight_decay':     1e-4,
    'n_epochs':         30,
    'patience':         7,
    'clip':             1.0,
    # Loss weights
    'sentiment_weight': 1.0,
    'emotion_weight':   1.0,   # raised from 0.5
    'recon_weight':     0.3,
    'checkpoint_path':  'best_model.pt',
    'results_path':     'results.txt',
}


# ── Sentiment metrics ──────────────────────────────────────────────────────────
def compute_sentiment_metrics(y_true, y_pred, to_print=False):
    mae  = np.mean(np.abs(y_pred - y_true))
    corr = np.corrcoef(y_pred, y_true)[0][1]

    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
    if len(non_zeros) > 0:
        binary_truth = (y_true[non_zeros] > 0)
        binary_preds = (y_pred[non_zeros] > 0)
        acc_2 = accuracy_score(binary_truth, binary_preds)
        f1_2  = f1_score(binary_truth, binary_preds, average='weighted')
    else:
        acc_2, f1_2 = 0.0, 0.0

    acc_nn = accuracy_score((y_true >= 0), (y_pred >= 0))
    acc_7  = accuracy_score(np.clip(np.round(y_true), -3, 3), np.clip(np.round(y_pred), -3, 3))

    if to_print:
        print(f"\n{'='*50}")
        print("SENTIMENT RESULTS")
        print(f"{'='*50}")
        print(f"MAE:                    {mae:.4f}")
        print(f"Pearson Correlation:    {corr:.4f}")
        print(f"Acc-7 (7-class):        {acc_7:.4f}")
        print(f"Acc-2 (pos/neg):        {acc_2:.4f}")
        print(f"F1 (pos/neg):           {f1_2:.4f}")
        print(f"Acc-2 (non-neg/neg):    {acc_nn:.4f}")
        if len(non_zeros) > 0:
            print("\nClassification Report (pos/neg):")
            print(classification_report((y_true[non_zeros] > 0), (y_pred[non_zeros] > 0), digits=5))

    return {'mae': mae, 'corr': corr, 'acc_7': acc_7, 'acc_2': acc_2, 'f1_2': f1_2, 'acc_nn': acc_nn}


# ── Emotion metrics ────────────────────────────────────────────────────────────
def compute_emotion_metrics(y_true, y_pred, to_print=False):
    """
    y_true, y_pred: (N, 6) arrays of emotion intensities
    Reports MAE per emotion and mean across all emotions.
    """
    mae_per_emotion = np.mean(np.abs(y_pred - y_true), axis=0)
    mae_mean = np.mean(mae_per_emotion)

    # Lower threshold — model predicts small values for rare emotions
    presence_true = (y_true > 0).astype(int)
    presence_pred = (y_pred > 0.15).astype(int)
    f1_per_emotion = []
    for i in range(6):
        if presence_true[:, i].sum() > 0:
            f1_per_emotion.append(f1_score(presence_true[:, i], presence_pred[:, i], zero_division=0))
        else:
            f1_per_emotion.append(0.0)
    f1_mean = np.mean(f1_per_emotion)

    if to_print:
        print(f"\n{'='*50}")
        print("EMOTION RESULTS")
        print(f"{'='*50}")
        print(f"{'Emotion':<12} {'MAE':>8} {'F1 (presence)':>15}")
        print("-" * 38)
        for i, name in enumerate(EMOTION_NAMES):
            print(f"{name:<12} {mae_per_emotion[i]:>8.4f} {f1_per_emotion[i]:>15.4f}")
        print("-" * 38)
        print(f"{'Mean':<12} {mae_mean:>8.4f} {f1_mean:>15.4f}")

    return {
        'emotion_mae': float(mae_mean),
        'emotion_f1':  float(f1_mean),
        'mae_per_emotion': [float(x) for x in mae_per_emotion],
        'f1_per_emotion':  [float(x) for x in f1_per_emotion],
    }


# ── Focal loss for emotions ────────────────────────────────────────────────────
def focal_emo_loss(pred, true, gamma=2.0):
    """
    Focal-style MSE loss that penalises missed detections more heavily.
    Scales the per-element MSE by how far the true value is from zero,
    so rare but present emotions receive a stronger gradient signal.
    """
    mse = (pred - true) ** 2
    weight = (true.abs() + 0.1) ** gamma
    return (mse * weight).mean()


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, sent_criterion, device, config):
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
        labels = batch['label'].to(device)    # (B, 7)

        sentiment_true = labels[:, 0:1]        # (B, 1)
        emotion_true   = labels[:, 1:]         # (B, 6)

        sentiment_pred, emotion_pred = model(bert, audio, visual, a_lens, v_lens)

        sent_loss  = sent_criterion(sentiment_pred, sentiment_true)
        emo_loss   = focal_emo_loss(emotion_pred, emotion_true)
        recon_loss = model.get_reconstruction_loss()

        loss = (config['sentiment_weight'] * sent_loss +
                config['emotion_weight']   * emo_loss  +
                config['recon_weight']     * recon_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()

        total_loss += loss.item()
        if (batch_i + 1) % 5 == 0 or (batch_i + 1) == n_batches:
            print(f"  Batch {batch_i+1}/{n_batches} | Loss: {total_loss/(batch_i+1):.4f}", end='\r')

    print()
    return total_loss / n_batches


def evaluate(model, loader, sent_criterion, device, config):
    model.eval()
    total_loss = 0
    sent_true_all, sent_pred_all = [], []
    emo_true_all,  emo_pred_all  = [], []

    with torch.no_grad():
        for batch in loader:
            bert   = batch['bert_feat'].to(device)
            visual = batch['visual'].to(device)
            audio  = batch['acoustic'].to(device)
            v_lens = batch['visual_lens'].to(device)
            a_lens = batch['acoustic_lens'].to(device)
            labels = batch['label'].to(device)

            sentiment_true = labels[:, 0:1]
            emotion_true   = labels[:, 1:]

            sentiment_pred, emotion_pred = model(bert, audio, visual, a_lens, v_lens)

            sent_loss = sent_criterion(sentiment_pred, sentiment_true)
            emo_loss  = focal_emo_loss(emotion_pred, emotion_true)
            loss = config['sentiment_weight'] * sent_loss + config['emotion_weight'] * emo_loss
            total_loss += loss.item()

            sent_pred_all.append(sentiment_pred.cpu().numpy())
            sent_true_all.append(sentiment_true.cpu().numpy())
            emo_pred_all.append(emotion_pred.cpu().numpy())
            emo_true_all.append(emotion_true.cpu().numpy())

    sent_true = np.concatenate(sent_true_all).squeeze()
    sent_pred = np.concatenate(sent_pred_all).squeeze()
    emo_true  = np.concatenate(emo_true_all,  axis=0)
    emo_pred  = np.concatenate(emo_pred_all,  axis=0)

    return total_loss / len(loader), sent_true, sent_pred, emo_true, emo_pred


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading data...")
    train_loader, dev_loader, test_loader = get_loaders(CONFIG['data_dir'], CONFIG['batch_size'])

    print("\nBuilding model...")
    model = AuraMetricsFusionModel(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    sent_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_dev_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    print(f"\n{'='*50}\nTRAINING START\n{'='*50}")

    for epoch in range(1, CONFIG['n_epochs'] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{CONFIG['n_epochs']}")
        print("-" * 30)

        train_loss = train_epoch(model, train_loader, optimizer, sent_criterion, device, CONFIG)

        dev_loss, dev_sent_true, dev_sent_pred, dev_emo_true, dev_emo_pred = evaluate(
            model, dev_loader, sent_criterion, device, CONFIG)

        dev_sent = compute_sentiment_metrics(dev_sent_true, dev_sent_pred)
        dev_emo  = compute_emotion_metrics(dev_emo_true, dev_emo_pred)
        epoch_time = time.time() - epoch_start

        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
              f"Sent Acc-2: {dev_sent['acc_2']:.4f} | Emo MAE: {dev_emo['emotion_mae']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        scheduler.step(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print(f"  >>> New best model saved (dev loss: {dev_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG['patience']:
                print("Early stopping.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(CONFIG['checkpoint_path'], weights_only=True))
    test_loss, test_sent_true, test_sent_pred, test_emo_true, test_emo_pred = evaluate(
        model, test_loader, sent_criterion, device, CONFIG)

    test_sent = compute_sentiment_metrics(test_sent_true, test_sent_pred, to_print=True)
    test_emo  = compute_emotion_metrics(test_emo_true, test_emo_pred, to_print=True)

    # ── Save results ───────────────────────────────────────────────────────────
    with open(CONFIG['results_path'], 'w') as f:
        f.write("AuraMetrics Multimodal Fusion Model — Test Results\n")
        f.write("=" * 50 + "\n")
        f.write("SENTIMENT\n")
        f.write(f"  MAE:                 {test_sent['mae']:.4f}\n")
        f.write(f"  Pearson Correlation: {test_sent['corr']:.4f}\n")
        f.write(f"  Acc-7:               {test_sent['acc_7']:.4f}\n")
        f.write(f"  Acc-2 (pos/neg):     {test_sent['acc_2']:.4f}\n")
        f.write(f"  F1 (pos/neg):        {test_sent['f1_2']:.4f}\n")
        f.write(f"  Acc-2 (non-neg/neg): {test_sent['acc_nn']:.4f}\n")
        f.write("\nEMOTION\n")
        f.write(f"  Mean MAE:            {test_emo['emotion_mae']:.4f}\n")
        f.write(f"  Mean F1 (presence):  {test_emo['emotion_f1']:.4f}\n")
        for i, name in enumerate(EMOTION_NAMES):
            f.write(f"  {name:<12} MAE: {test_emo['mae_per_emotion'][i]:.4f}  F1: {test_emo['f1_per_emotion'][i]:.4f}\n")
        f.write(f"\nTraining time: {total_time/60:.1f} minutes\n")

    # ── Run history ────────────────────────────────────────────────────────────
    run_record = {
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(train_loader.dataset),
        'epochs_run':    epoch,
        'hidden_dim':    CONFIG['hidden_dim'],
        'batch_size':    CONFIG['batch_size'],
        'sent_mae':      round(float(test_sent['mae']),    4),
        'sent_corr':     round(float(test_sent['corr']),   4),
        'sent_acc_7':    round(float(test_sent['acc_7']),  4),
        'sent_acc_2':    round(float(test_sent['acc_2']),  4),
        'sent_f1':       round(float(test_sent['f1_2']),   4),
        'emo_mae':       round(float(test_emo['emotion_mae']), 4),
        'emo_f1':        round(float(test_emo['emotion_f1']),  4),
        'time_mins':     round(total_time / 60, 1),
    }

    history_path = 'run_history.json'
    history = {'runs': [], 'best': None}
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except:
            history = {'runs': [], 'best': None}

    history['runs'].append(run_record)
    if history['best'] is None or run_record['sent_acc_2'] > history['best']['sent_acc_2']:
        history['best'] = run_record
        print(f"\n*** NEW BEST! Sent Acc-2: {run_record['sent_acc_2']}")
    else:
        print(f"\nBest remains: Acc-2: {history['best']['sent_acc_2']} ({history['best']['timestamp']})")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}\nRUN HISTORY\n{'='*70}")
    print(f"{'#':<4} {'Date':<20} {'Samples':<8} {'Acc-2':<8} {'F1':<8} {'Sent MAE':<10} {'Emo MAE':<10}")
    print("-" * 70)
    for i, run in enumerate(history['runs'], 1):
        marker = " ***" if run == history['best'] else ""
        print(f"{i:<4} {run['timestamp']:<20} {run['train_samples']:<8} "
              f"{run['sent_acc_2']:<8} {run['sent_f1']:<8} {run['sent_mae']:<10} {run['emo_mae']:<10}{marker}")

    print(f"\nResults saved to {CONFIG['results_path']}")
    print(f"Run history saved to {history_path}")


if __name__ == '__main__':
    main()