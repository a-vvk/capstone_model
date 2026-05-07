import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: query from one modality, key/value from another.
    Allows each modality to attend to relevant parts of other modalities.
    """
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
    """
    Encodes a variable-length sequence of features into a fixed-size representation.
    Uses a bidirectional LSTM followed by mean pooling.
    """
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
    """
    Learns to weight each modality's contribution based on quality.
    Takes all three encoded modalities and produces a softmax weight
    for each, allowing the model to dynamically downweight noisy or
    missing modalities at inference time.
    """
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
        weights = self.gate(combined)  # (B, 3)
        return weights


class AuraMetricsFusionModel(nn.Module):
    """
    AuraMetrics Custom Multimodal Fusion Model (v2).

    Architecture:
    - Text: DistilBERT CLS embedding (pre-extracted, 768-dim) -> projection
    - Audio: Bidirectional LSTM encoder
    - Visual: Bidirectional LSTM encoder
    - Modality Gate: learned dynamic weighting of each modality
    - Fusion: Cross-modal attention between all modality pairs
    - Modality dropout: aggressive pattern-based dropout during training
    - Reconstruction loss: predicts audio/visual from text to strengthen shared reps
    - Output: Regression head for continuous sentiment prediction [-3, +3]

    Based on principles from MISA (Hazarika et al., 2020) with improvements
    from recent missing-modality robustness literature.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        hidden = config['hidden_dim']
        dropout = config['dropout']

        # ── Text encoder (DistilBERT features already extracted) ──────────────
        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden)
        )

        # ── Audio encoder ─────────────────────────────────────────────────────
        self.audio_encoder = ModalityEncoder(config['audio_dim'], hidden, dropout)

        # ── Visual encoder ────────────────────────────────────────────────────
        self.visual_encoder = ModalityEncoder(config['visual_dim'], hidden, dropout)

        # ── Cross-modal attention ─────────────────────────────────────────────
        self.text_audio_attn   = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.text_visual_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.audio_text_attn   = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.visual_text_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)

        # ── Modality gate ─────────────────────────────────────────────────────
        self.modality_gate = ModalityGate(hidden)

        # ── Fusion layer ──────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.output = nn.Linear(hidden, 1)

        # ── Reconstruction heads (predict audio/visual from text) ─────────────
        self.reconstruct_audio = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.reconstruct_visual = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # ── Modality dropout probability ──────────────────────────────────────
        self.modality_dropout = config.get('modality_dropout', 0.2)

        # ── Store intermediate representations for reconstruction loss ────────
        self._t_encoded = None
        self._a_encoded = None
        self._v_encoded = None

    def forward(self, text_feat, audio, visual, audio_lengths, visual_lengths):
        """
        Args:
            text_feat:      (B, 768) DistilBERT CLS embeddings
            audio:          (B, T_a, audio_dim) padded audio sequences
            visual:         (B, T_v, visual_dim) padded visual sequences
            audio_lengths:  (B,) actual lengths of audio sequences
            visual_lengths: (B,) actual lengths of visual sequences
        """
        # ── Encode each modality ──────────────────────────────────────────────
        t = self.text_proj(text_feat)
        a = self.audio_encoder(audio, audio_lengths)
        v = self.visual_encoder(visual, visual_lengths)

        # Store clean encodings for reconstruction loss (before dropout)
        self._t_encoded = t.detach()
        self._a_encoded = a.detach()
        self._v_encoded = v.detach()

        # ── Aggressive modality dropout (training only) ───────────────────────
        if self.training and self.modality_dropout > 0:
            batch_size = t.size(0)
            device = t.device
            pattern = torch.rand(batch_size, device=device)

            for b in range(batch_size):
                r = pattern[b].item()
                if r < 0.10:        # drop audio only
                    a[b] = torch.zeros_like(a[b])
                elif r < 0.20:      # drop visual only
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.30:      # drop both audio + visual (text only)
                    a[b] = torch.zeros_like(a[b])
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.35:      # drop text only
                    t[b] = torch.zeros_like(t[b])
                elif r < 0.40:      # add noise to all
                    t[b] = t[b] + torch.randn_like(t[b]) * 0.1
                    a[b] = a[b] + torch.randn_like(a[b]) * 0.1
                    v[b] = v[b] + torch.randn_like(v[b]) * 0.1
                # else: keep all (60% of the time)

        # ── Cross-modal attention ─────────────────────────────────────────────
        t_enriched = self.text_audio_attn(t, a)
        t_enriched = self.text_visual_attn(t_enriched, v)
        a_enriched = self.audio_text_attn(a, t)
        v_enriched = self.visual_text_attn(v, t)

        # ── Modality gating ───────────────────────────────────────────────────
        weights = self.modality_gate(t_enriched, a_enriched, v_enriched)  # (B, 3)
        t_weighted = t_enriched * weights[:, 0:1]
        a_weighted = a_enriched * weights[:, 1:2]
        v_weighted = v_enriched * weights[:, 2:3]

        # ── Concatenate and fuse ──────────────────────────────────────────────
        fused = torch.cat([t_weighted, a_weighted, v_weighted], dim=-1)
        fused = self.fusion(fused)

        # ── Predict sentiment score ───────────────────────────────────────────
        out = self.output(fused)
        return out

    def get_reconstruction_loss(self):
        """
        Predict audio and visual representations from text.
        Forces text encoder to learn representations that contain enough
        information to reconstruct the other modalities, which strengthens
        cross-modal alignment and helps when modalities are missing.
        """
        if self._t_encoded is None:
            return torch.tensor(0.0)

        a_pred = self.reconstruct_audio(self._t_encoded)
        v_pred = self.reconstruct_visual(self._t_encoded)
        loss = F.mse_loss(a_pred, self._a_encoded) + F.mse_loss(v_pred, self._v_encoded)
        return loss / 2.0