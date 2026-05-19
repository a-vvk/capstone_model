import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Lets one modality attend to another using multi-head attention."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(1)
        return self.norm(query + self.dropout(out))


class ModalityEncoder(nn.Module):
    """Encodes a variable-length sequence into a fixed-size vector using a BiLSTM."""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.proj    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # mean pool over actual sequence length
        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        out  = (out * mask).sum(1) / mask.sum(1)

        return self.dropout(self.norm(F.relu(self.proj(out))))


class ModalityGate(nn.Module):
    """Learns to weight each modality based on how useful it is for the current sample."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, t, a, v):
        return self.gate(torch.cat([t, a, v], dim=-1))


class AuraMetricsFusionModel(nn.Module):
    """
    AuraMetrics multimodal fusion model.

    Takes text (DistilBERT CLS embeddings), audio (COVAREP), and visual (OpenFace 2.0)
    and predicts:
      - sentiment score in [-3, +3]
      - 6 emotion intensities in [0, 3] (happiness, sadness, anger, fear, disgust, surprise)

    Key design choices:
      - Cross-modal attention so each modality can pick up info from the others
      - Learned modality gate to handle missing/noisy inputs at inference time
      - Modality dropout during training so the model doesn't rely too heavily on any one input
      - Auxiliary reconstruction loss to strengthen cross-modal representations
    """

    def __init__(self, config):
        super().__init__()
        hidden  = config['hidden_dim']
        dropout = config['dropout']

        # text: DistilBERT features already extracted, just project down to hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden)
        )

        # audio and visual: variable-length sequences encoded by BiLSTM
        self.audio_encoder  = ModalityEncoder(config['audio_dim'],  hidden, dropout)
        self.visual_encoder = ModalityEncoder(config['visual_dim'], hidden, dropout)

        # cross-modal attention: text attends to audio/visual, audio/visual attend to text
        self.text_audio_attn  = CrossModalAttention(hidden, dropout=dropout)
        self.text_visual_attn = CrossModalAttention(hidden, dropout=dropout)
        self.audio_text_attn  = CrossModalAttention(hidden, dropout=dropout)
        self.visual_text_attn = CrossModalAttention(hidden, dropout=dropout)

        # modality gate: learned weighting of each modality's contribution
        self.modality_gate = ModalityGate(hidden)

        # fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),    nn.ReLU(), nn.Dropout(dropout),
        )

        # output heads
        self.sentiment_head = nn.Linear(hidden, 1)
        self.emotion_head   = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 6),
            nn.Sigmoid()
        )

        # reconstruction heads: predict audio/visual from text
        self.reconstruct_audio  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.reconstruct_visual = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        self.modality_dropout = config.get('modality_dropout', 0.2)
        self._t_encoded = None
        self._a_encoded = None
        self._v_encoded = None

    def forward(self, text_feat, audio, visual, audio_lengths, visual_lengths):
        t = self.text_proj(text_feat)
        a = self.audio_encoder(audio, audio_lengths)
        v = self.visual_encoder(visual, visual_lengths)

        # save clean encodings for reconstruction loss before applying dropout
        self._t_encoded = t.detach()
        self._a_encoded = a.detach()
        self._v_encoded = v.detach()

        # modality dropout during training
        if self.training and self.modality_dropout > 0:
            r = torch.rand(t.size(0), device=t.device)
            for b in range(t.size(0)):
                if   r[b] < 0.10: a[b].zero_()
                elif r[b] < 0.20: v[b].zero_()
                elif r[b] < 0.30: a[b].zero_(); v[b].zero_()
                elif r[b] < 0.35: t[b].zero_()
                elif r[b] < 0.40:
                    t[b] = t[b] + torch.randn_like(t[b]) * 0.1
                    a[b] = a[b] + torch.randn_like(a[b]) * 0.1
                    v[b] = v[b] + torch.randn_like(v[b]) * 0.1

        # cross-modal attention
        t = self.text_visual_attn(self.text_audio_attn(t, a), v)
        a = self.audio_text_attn(a, t)
        v = self.visual_text_attn(v, t)

        # gate + fuse
        w     = self.modality_gate(t, a, v)
        fused = self.fusion(torch.cat([t * w[:, 0:1], a * w[:, 1:2], v * w[:, 2:3]], dim=-1))

        sentiment = self.sentiment_head(fused)
        emotion   = self.emotion_head(fused) * 3.0  # scale to [0, 3]

        return sentiment, emotion

    def get_reconstruction_loss(self):
        if self._t_encoded is None:
            return torch.tensor(0.0)
        a_pred = self.reconstruct_audio(self._t_encoded)
        v_pred = self.reconstruct_visual(self._t_encoded)
        return (F.mse_loss(a_pred, self._a_encoded) + F.mse_loss(v_pred, self._v_encoded)) / 2.0