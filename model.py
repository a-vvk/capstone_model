import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    
 # Cross-modal attention: query from one modality, key/value from another.
 # Allows each modality to attend to relevant parts of other modalities.

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: (B, dim), key_value: (B, dim)
        # Expand to sequence length 1 for attention
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(1)
        return self.norm(query + self.dropout(out))

class ModalityEncoder(nn.Module):
    
    # Encodes a variable-length sequence of features into a fixed-size representation.
    # Uses a bidirectional LSTM followed by mean pooling.

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
        # x: (B, T, input_dim), lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Mean pool over time
        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        out = (out * mask).sum(1) / mask.sum(1)
        out = self.norm(F.relu(self.proj(out)))
        return self.dropout(out)


class AuraMetricsFusionModel(nn.Module):

    # AuraMetrics Custom Multimodal Fusion Model.

    # Architecture:
    # - Text: DistilBERT CLS embedding (pre-extracted, 768-dim) -> projection
    # - Audio: Bidirectional LSTM encoder
    # - Visual: Bidirectional LSTM encoder
    # - Fusion: Cross-modal attention between all modality pairs
    # - Modality dropout: randomly zeros out modalities during training
    #   to ensure robustness when modalities are missing at inference time
    # - Output: Regression head for continuous sentiment prediction [-3, +3]

    # Based on principles from MISA - lightweight architecture designed for deployment on limited hardware.

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
        # Text attends to audio and visual
        self.text_audio_attn   = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        self.text_visual_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        # Audio attends to text
        self.audio_text_attn   = CrossModalAttention(hidden, num_heads=4, dropout=dropout)
        # Visual attends to text
        self.visual_text_attn  = CrossModalAttention(hidden, num_heads=4, dropout=dropout)

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

        # ── Modality dropout probabilities ────────────────────────────────────
        self.modality_dropout = config.get('modality_dropout', 0.2)

    def forward(self, text_feat, audio, visual, audio_lengths, visual_lengths):
        
        # Args:
        #     text_feat:      (B, 768) DistilBERT CLS embeddings
        #     audio:          (B, T_a, audio_dim) padded audio sequences
        #     visual:         (B, T_v, visual_dim) padded visual sequences
        #     audio_lengths:  (B,) actual lengths of audio sequences
        #     visual_lengths: (B,) actual lengths of visual sequences

        # ── Encode each modality ──────────────────────────────────────────────
        t = self.text_proj(text_feat)           # (B, hidden)
        a = self.audio_encoder(audio, audio_lengths)   # (B, hidden)
        v = self.visual_encoder(visual, visual_lengths) # (B, hidden)

        # ── Modality dropout (training only) ──────────────────────────────────
        # Randomly zero out entire modalities to simulate missing inputs

        if self.training and self.modality_dropout > 0:
            batch_size = t.size(0)
            device = t.device
            p = self.modality_dropout
            
            # For each sample, randomly choose a dropout pattern:
            # 0 = keep all (60%), 1 = drop audio (10%), 2 = drop visual (10%),
            # 3 = drop both audio+visual (10%), 4 = drop text (5%), 5 = noise all (5%)
            pattern = torch.rand(batch_size, device=device)
            
            for b in range(batch_size):
                r = pattern[b].item()
                if r < 0.10:        # drop audio
                    a[b] = torch.zeros_like(a[b])
                elif r < 0.20:      # drop visual
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.30:      # drop both audio + visual (text only)
                    a[b] = torch.zeros_like(a[b])
                    v[b] = torch.zeros_like(v[b])
                elif r < 0.35:      # drop text
                    t[b] = torch.zeros_like(t[b])
                elif r < 0.40:      # add noise to all
                    t[b] += torch.randn_like(t[b]) * 0.1
                    a[b] += torch.randn_like(a[b]) * 0.1
                    v[b] += torch.randn_like(v[b]) * 0.1
                # else: keep all (60% of the time)

        # ── Cross-modal attention ─────────────────────────────────────────────
        t_enriched = self.text_audio_attn(t, a)   # text attends to audio
        t_enriched = self.text_visual_attn(t_enriched, v)  # text attends to visual
        a_enriched = self.audio_text_attn(a, t)   # audio attends to text
        v_enriched = self.visual_text_attn(v, t)  # visual attends to text

        # ── Concatenate and fuse ──────────────────────────────────────────────
        fused = torch.cat([t_enriched, a_enriched, v_enriched], dim=-1)  # (B, hidden*3)
        fused = self.fusion(fused)

        # ── Predict sentiment score ───────────────────────────────────────────
        out = self.output(fused)  # (B, 1)
        return out