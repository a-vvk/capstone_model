import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MOSEIDataset(Dataset):
    """
    Dataset loader for CMU-MOSEI with pre-extracted DistilBERT features.

    Each sample contains:
    - words:        word ID sequence (not used directly, kept for compatibility)
    - visual:       OpenFace 2.0 facial action unit sequence (T_v, 35)
    - acoustic:     COVAREP acoustic feature sequence (T_a, 74)
    - actual_words: list of word strings
    - bert_feat:    DistilBERT CLS embedding (768,)
    - label:        sentiment score (1,) averaged across annotators
    """

    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} samples from {pkl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (words, visual, acoustic, actual_words, bert_feat), label, vid = self.data[idx]

        # Handle label shape — take mean sentiment score
        if isinstance(label, np.ndarray):
            if label.ndim == 2:
                sentiment = float(np.nanmean(label[:, 0]))
            elif label.ndim == 1:
                sentiment = float(np.nanmean(label))
            else:
                sentiment = float(label)
        else:
            sentiment = float(label)

        return {
            'bert_feat': torch.tensor(bert_feat, dtype=torch.float32),
            'visual':    torch.tensor(visual,    dtype=torch.float32),
            'acoustic':  torch.tensor(acoustic,  dtype=torch.float32),
            'label':     torch.tensor([sentiment], dtype=torch.float32),
            'vid':       vid
        }


def collate_fn(batch):
    """
    Custom collate function to pad variable-length sequences.
    Returns padded tensors and actual lengths for packing in LSTM.
    """
    bert_feats  = torch.stack([b['bert_feat'] for b in batch])
    labels      = torch.stack([b['label']     for b in batch])

    # Pad visual sequences
    visual_seqs = [b['visual'] for b in batch]
    visual_lens = torch.tensor([v.size(0) for v in visual_seqs], dtype=torch.long)
    visual_pad  = pad_sequence(visual_seqs, batch_first=True)

    # Pad acoustic sequences
    acoustic_seqs = [b['acoustic'] for b in batch]
    acoustic_lens = torch.tensor([a.size(0) for a in acoustic_seqs], dtype=torch.long)
    acoustic_pad  = pad_sequence(acoustic_seqs, batch_first=True)

    # Clamp lengths to at least 1 to avoid pack_padded_sequence errors
    visual_lens  = visual_lens.clamp(min=1)
    acoustic_lens = acoustic_lens.clamp(min=1)

    return {
        'bert_feat':     bert_feats,
        'visual':        visual_pad,
        'acoustic':      acoustic_pad,
        'visual_lens':   visual_lens,
        'acoustic_lens': acoustic_lens,
        'label':         labels,
    }


def get_loaders(data_dir, batch_size=16, num_workers=0):
    train_set = MOSEIDataset(f'{data_dir}/train_bert.pkl')
    dev_set   = MOSEIDataset(f'{data_dir}/dev_bert.pkl')
    test_set  = MOSEIDataset(f'{data_dir}/test_bert.pkl')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    dev_loader   = DataLoader(dev_set,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, dev_loader, test_loader