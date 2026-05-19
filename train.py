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
    'sentiment_weight': 1.0,
    'emotion_weight':   1.0,
    'recon_weight':     0.3,
    'checkpoint_path':  'best_model.pt',
    'results_path':     'results.txt',
}


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
    acc_7  = accuracy_score(
        np.clip(np.round(y_true), -3, 3),
        np.clip(np.round(y_pred), -3, 3)
    )

    if to_print:
        print(f"\n{'='*50}\nSENTIMENT RESULTS\n{'='*50}")
        print(f"MAE:                    {mae:.4f}")
        print(f"Pearson Correlation:    {corr:.4f}")
        print(f"Acc-7 (7-class):        {acc_7:.4f}")
        print(f"Acc-2 (pos/neg):        {acc_2:.4f}")
        print(f"F1 (pos/neg):           {f1_2:.4f}")
        print(f"Acc-2 (non-neg/neg):    {acc_nn:.4f}")
        if len(non_zeros) > 0:
            print("\nClassification Report (pos/neg):")
            print(classification_report(binary_truth, binary_preds, digits=5))

    return {'mae': mae, 'corr': corr, 'acc_7': acc_7, 'acc_2': acc_2, 'f1_2': f1_2, 'acc_nn': acc_nn}


def compute_emotion_metrics(y_true, y_pred, to_print=False):
    mae_per = np.mean(np.abs(y_pred - y_true), axis=0)
    mae_mean = np.mean(mae_per)

    # use 0.15 threshold — model predicts small values for rare emotions
    presence_true = (y_true > 0).astype(int)
    presence_pred = (y_pred > 0.15).astype(int)

    f1_per = []
    for i in range(6):
        if presence_true[:, i].sum() > 0:
            f1_per.append(f1_score(presence_true[:, i], presence_pred[:, i], zero_division=0))
        else:
            f1_per.append(0.0)
    f1_mean = np.mean(f1_per)

    if to_print:
        print(f"\n{'='*50}\nEMOTION RESULTS\n{'='*50}")
        print(f"{'Emotion':<12} {'MAE':>8} {'F1 (presence)':>15}")
        print("-" * 38)
        for i, name in enumerate(EMOTION_NAMES):
            print(f"{name:<12} {mae_per[i]:>8.4f} {f1_per[i]:>15.4f}")
        print("-" * 38)
        print(f"{'Mean':<12} {mae_mean:>8.4f} {f1_mean:>15.4f}")

    return {
        'emotion_mae':     float(mae_mean),
        'emotion_f1':      float(f1_mean),
        'mae_per_emotion': [float(x) for x in mae_per],
        'f1_per_emotion':  [float(x) for x in f1_per],
    }


def focal_emo_loss(pred, true, gamma=2.0):
    """
    Focal-style MSE for emotion labels.
    Scales the loss by how far the true value is from zero so rare
    but present emotions get a stronger training signal.
    """
    mse    = (pred - true) ** 2
    weight = (true.abs() + 0.1) ** gamma
    return (mse * weight).mean()


def train_epoch(model, loader, optimizer, sent_criterion, device, config):
    model.train()
    total_loss = 0

    for batch_i, batch in enumerate(loader):
        optimizer.zero_grad()

        bert   = batch['bert_feat'].to(device)
        visual = batch['visual'].to(device)
        audio  = batch['acoustic'].to(device)
        v_lens = batch['visual_lens'].to(device)
        a_lens = batch['acoustic_lens'].to(device)
        labels = batch['label'].to(device)

        sent_true = labels[:, 0:1]
        emo_true  = labels[:, 1:]

        sent_pred, emo_pred = model(bert, audio, visual, a_lens, v_lens)

        sent_loss  = sent_criterion(sent_pred, sent_true)
        emo_loss   = focal_emo_loss(emo_pred, emo_true)
        recon_loss = model.get_reconstruction_loss()

        loss = (config['sentiment_weight'] * sent_loss +
                config['emotion_weight']   * emo_loss  +
                config['recon_weight']     * recon_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()

        total_loss += loss.item()
        n = len(loader)
        if (batch_i + 1) % 5 == 0 or (batch_i + 1) == n:
            print(f"  Batch {batch_i+1}/{n} | Loss: {total_loss/(batch_i+1):.4f}", end='\r')

    print()
    return total_loss / len(loader)


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

            sent_true = labels[:, 0:1]
            emo_true  = labels[:, 1:]

            sent_pred, emo_pred = model(bert, audio, visual, a_lens, v_lens)

            loss = (config['sentiment_weight'] * sent_criterion(sent_pred, sent_true) +
                    config['emotion_weight']   * focal_emo_loss(emo_pred, emo_true))
            total_loss += loss.item()

            sent_pred_all.append(sent_pred.cpu().numpy())
            sent_true_all.append(sent_true.cpu().numpy())
            emo_pred_all.append(emo_pred.cpu().numpy())
            emo_true_all.append(emo_true.cpu().numpy())

    sent_true = np.concatenate(sent_true_all).squeeze()
    sent_pred = np.concatenate(sent_pred_all).squeeze()
    emo_true  = np.concatenate(emo_true_all,  axis=0)
    emo_pred  = np.concatenate(emo_pred_all,  axis=0)

    return total_loss / len(loader), sent_true, sent_pred, emo_true, emo_pred


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, dev_loader, test_loader = get_loaders(CONFIG['data_dir'], CONFIG['batch_size'])

    model = AuraMetricsFusionModel(CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    sent_criterion = nn.MSELoss()
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=CONFIG['learning_rate'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_dev_loss    = float('inf')
    patience_counter = 0
    start_time       = time.time()

    print(f"\n{'='*50}\nTRAINING START\n{'='*50}")

    for epoch in range(1, CONFIG['n_epochs'] + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{CONFIG['n_epochs']}\n{'-'*30}")

        train_loss = train_epoch(model, train_loader, optimizer, sent_criterion, device, CONFIG)
        dev_loss, dev_st, dev_sp, dev_et, dev_ep = evaluate(
            model, dev_loader, sent_criterion, device, CONFIG)

        dev_sent = compute_sentiment_metrics(dev_st, dev_sp)
        dev_emo  = compute_emotion_metrics(dev_et, dev_ep)

        print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
              f"Sent Acc-2: {dev_sent['acc_2']:.4f} | Emo MAE: {dev_emo['emotion_mae']:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        scheduler.step(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss    = dev_loss
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

    # evaluate on test set using best model
    model.load_state_dict(torch.load(CONFIG['checkpoint_path'], weights_only=True))
    _, test_st, test_sp, test_et, test_ep = evaluate(
        model, test_loader, sent_criterion, device, CONFIG)

    test_sent = compute_sentiment_metrics(test_st, test_sp, to_print=True)
    test_emo  = compute_emotion_metrics(test_et, test_ep, to_print=True)

    # save results
    with open(CONFIG['results_path'], 'w') as f:
        f.write("AuraMetrics — Test Results\n" + "="*50 + "\n")
        f.write("SENTIMENT\n")
        for k, v in test_sent.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nEMOTION\n")
        f.write(f"  mean_mae: {test_emo['emotion_mae']:.4f}\n")
        f.write(f"  mean_f1:  {test_emo['emotion_f1']:.4f}\n")
        for i, name in enumerate(EMOTION_NAMES):
            f.write(f"  {name:<12} mae: {test_emo['mae_per_emotion'][i]:.4f}  "
                    f"f1: {test_emo['f1_per_emotion'][i]:.4f}\n")
        f.write(f"\ntime: {total_time/60:.1f} mins\n")

    # run history
    run = {
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(train_loader.dataset),
        'epochs_run':    epoch,
        'hidden_dim':    CONFIG['hidden_dim'],
        'batch_size':    CONFIG['batch_size'],
        'sent_acc_2':    round(float(test_sent['acc_2']),       4),
        'sent_f1':       round(float(test_sent['f1_2']),        4),
        'sent_mae':      round(float(test_sent['mae']),         4),
        'sent_corr':     round(float(test_sent['corr']),        4),
        'sent_acc_7':    round(float(test_sent['acc_7']),       4),
        'emo_mae':       round(float(test_emo['emotion_mae']),  4),
        'emo_f1':        round(float(test_emo['emotion_f1']),   4),
        'time_mins':     round(total_time / 60, 1),
    }

    history_path = 'run_history.json'
    history = {'runs': [], 'best': None}
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
        except:
            pass

    history['runs'].append(run)
    if history['best'] is None or run['sent_acc_2'] > history['best']['sent_acc_2']:
        history['best'] = run
        print(f"\n*** NEW BEST! Acc-2: {run['sent_acc_2']}")
    else:
        print(f"\nBest: {history['best']['sent_acc_2']} ({history['best']['timestamp']})")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # print run history table
    print(f"\n{'='*70}\nRUN HISTORY\n{'='*70}")
    print(f"{'#':<4} {'Date':<20} {'Samples':<8} {'Acc-2':<8} {'F1':<8} {'MAE':<8} {'Emo F1':<8}")
    print("-" * 70)
    for i, r in enumerate(history['runs'], 1):
        mark = " ***" if r == history['best'] else ""
        print(f"{i:<4} {r['timestamp']:<20} {r['train_samples']:<8} "
              f"{r['sent_acc_2']:<8} {r['sent_f1']:<8} {r['sent_mae']:<8} {r['emo_f1']:<8}{mark}")


if __name__ == '__main__':
    main()