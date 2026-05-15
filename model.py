import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(1)
        return self.norm(query + self.dropout(out))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        out = (out * mask).sum(1) / mask.sum(1)
        out = self.norm(F.relu(self.proj(out)))
        return self.dropout(out)


class ModalityGate(nn.Module):
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
        combined = torch.cat([t, a, v], dim=-1)
        return self.gate(combined)


class AuraMetricsFusionModel(nn.Module):
    """
    AuraMetrics Custom Multimodal Fusion Model (v3) — Multi-task.

    Outputs:
    - sentiment_pred: (B, 1) continuous score [-3, +3]
    - emotion_pred:   (B, 6) intensity scores for
                      [happiness, sadness, anger, fear, disgust, surprise]
                      each in [0, 3] via sigmoid * 3 scaling

    Multi-task learning improves both tasks — shared representations
    learn richer features when jointly optimised for sentiment and emotion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config['hidden_dim']
        dropout = config['dropout']

        # ── Encoders ──────────────────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden)
        )
        self.audio_encoder  = ModalityEncoder(config['audio_dim'],  hidden, dropout)
        self.visual_encoder = ModalityEncoder(config['visual_dim'], hidden, dropout)

        # ── Cross-modal attention ─────────────────────────────────────────────
        self.text_audio_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.text_visual_attn = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.audio_text_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.visual_text_attn = CrossModalAttention(hidden, num_heads=4, dropout=dropout)

        # ── Modality gate ─────────────────────────────────────────────────────
        self.modality_gate = ModalityGate(hidden)

        # ── Shared fusion ─────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Task heads ────────────────────────────────────────────────────────
        # Sentiment: continuous regression [-3, +3]
        self.sentiment_head = nn.Linear(hidden, 1)

        # Emotion: 6 independent intensities [0, 3]
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 6),
            nn.Sigmoid()   # outputs [0, 1], scaled to [0, 3] in forward
        )

        # ── Reconstruction heads ──────────────────────────────────────────────
        self.reconstruct_audio  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.reconstruct_visual = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.modality_dropout = config.get('modality_dropout', 0.2)
        self._t_encoded = None
        self._a_encoded = None
        self._v_encoded = None

    def forward(self, text_feat, audio, visual, audio_lengths, visual_lengths):
        # ── Encode ────────────────────────────────────────────────────────────
        t = self.text_proj(text_feat)
        a = self.audio_encoder(audio, audio_lengths)
        v = self.visual_encoder(visual, visual_lengths)

        self._t_encoded = t.detach()
        self._a_encoded = a.detach()
        self._v_encoded = v.detach()

        # ── Modality dropout (training) ───────────────────────────────────────
        if self.training and self.modality_dropout > 0:
            batch_size = t.size(0)
            pattern = torch.rand(batch_size, device=t.device)
            for b in range(batch_size):
                r = pattern[b].item()
                if r < 0.10:
                    a[b] = torch.zeros_like(a[b])
                elif r < 0.20:
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.30:
                    a[b] = torch.zeros_like(a[b])
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.35:
                    t[b] = torch.zeros_like(t[b])
                elif r < 0.40:
                    t[b] = t[b] + torch.randn_like(t[b]) * 0.1
                    a[b] = a[b] + torch.randn_like(a[b]) * 0.1
                    v[b] = v[b] + torch.randn_like(v[b]) * 0.1

        # ── Cross-modal attention ─────────────────────────────────────────────
        t_e = self.text_visual_attn(self.text_audio_attn(t, a), v)
        a_e = self.audio_text_attn(a, t)
        v_e = self.visual_text_attn(v, t)

        # ── Gate + fuse ───────────────────────────────────────────────────────
        w = self.modality_gate(t_e, a_e, v_e)
        fused = self.fusion(torch.cat([t_e * w[:, 0:1], a_e * w[:, 1:2], v_e * w[:, 2:3]], dim=-1))

        # ── Predict ───────────────────────────────────────────────────────────
        sentiment = self.sentiment_head(fused)             # (B, 1)
        emotion   = self.emotion_head(fused) * 3.0         # (B, 6) scaled to [0, 3]

        return sentiment, emotion

    def get_reconstruction_loss(self):
        if self._t_encoded is None:
            return torch.tensor(0.0)
        a_pred = self.reconstruct_audio(self._t_encoded)
        v_pred = self.reconstruct_visual(self._t_encoded)
        return (F.mse_loss(a_pred, self._a_encoded) + F.mse_loss(v_pred, self._v_encoded)) / 2.0