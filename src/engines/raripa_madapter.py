import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engines.raripa_combined import RARIPACombined

# Default mAda3 setting from the paper (Section 4.2.1):
# 3 layers, each with k=3, s=2, p=1 (8x compression overall).
# (For mAda1, pass num_layers=1, kernel=8, stride=8, padding=4.)
KERNEL, STRIDE, PADDING = 3, 2, 1


class LitAdapter(nn.Module):
    """Lightweight adapter placed before M-Adapter (Section 4.2.1, Footnote 3;
    Gállego et al., 2021): LayerNorm + 2 linear layers with ReLU + residual.
    """

    def __init__(self, dim=1024, bottleneck_dim=1024, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.fc2(F.relu(self.fc1(self.norm(x)))))


class MAdapterLayer(nn.Module):
    """Single M-Adapter block with Multi-head Pooled Self-Attention (MPSA)
    and pooled residual connection (Section 3.2, Figures 1 & 2).
    """

    def __init__(
        self,
        dim=1024,
        heads=16,
        ffn_dim=4096,
        kernel=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dropout=0.1,
        use_mpsa=True,
    ):
        super().__init__()
        self.heads = heads
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.use_mpsa = use_mpsa

        self.pool_x = nn.Conv1d(dim, dim, kernel, stride, padding)
        if use_mpsa:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.pool_q, self.pool_k, self.pool_v = (
                nn.Conv1d(dim, dim, kernel, stride, padding) for _ in range(3)
            )
            self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths=None):  # (B, L, D) -> (B, L', D)
        if lengths is not None:
            positions = torch.arange(x.size(1), device=x.device)
            pad_mask = positions[None, :] < lengths[:, None]
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)

        pool = lambda conv, t: conv(t.transpose(1, 2)).transpose(1, 2)
        x_pooled = pool(self.pool_x, x)

        if lengths is not None:
            lengths = (lengths + 2 * self.padding - self.kernel) // self.stride + 1
            lengths = lengths.clamp(min=0, max=x_pooled.size(1))

        if self.use_mpsa:
            pools = (self.pool_q, self.pool_k, self.pool_v)
            q, k, v = (pool(p, t) for p, t in zip(pools, self.qkv(x).chunk(3, dim=-1)))

            attn_mask = None
            if lengths is not None:
                positions = torch.arange(x_pooled.size(1), device=x.device)
                # Clamp to min=1 for attention mask so fully empty rows don't produce NaN in softmax
                valid_lengths = lengths.clamp(min=1)
                attn_mask = (positions[None, :] < valid_lengths[:, None])[
                    :, None, None, :
                ]

            q, k, v = (
                t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v)
            )
            h = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            h = h.transpose(1, 2).flatten(2)
            x = self.norm1(x_pooled + self.dropout(self.out(h)))
        else:
            x = self.norm1(x_pooled)

        x = self.norm2(x + self.ffn(x))
        return (x, lengths) if lengths is not None else x


class MAdapter(nn.Module):
    """Full M-Adapter bridge: LitAdapter -> N x MAdapterLayer -> Linear to d_model."""

    def __init__(
        self,
        in_dim=1024,
        out_dim=1472,
        # Reduced from 3 layers (8x -> 6.25 Hz) to 2 layers (4x -> 12.5 Hz) to
        # preserve phoneme-level temporal resolution for byte-level ByT5.
        num_layers=2,
        heads=16,
        # NOTE: Shrunk from 4096 to 1024 for small datasets like Nexdata (616 samples);
        # can be scaled back up (e.g. 2048 or 4096) for larger datasets like FLEURS/FSC.
        ffn_dim=1024,
        kernel=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dropout=0.1,
        use_mpsa=True,
    ):
        super().__init__()
        self.lit_adapter = LitAdapter(in_dim, in_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                MAdapterLayer(
                    dim=in_dim,
                    heads=heads,
                    ffn_dim=ffn_dim,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    use_mpsa=use_mpsa,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(in_dim, out_dim)

        # Near-zero initialization for stable training into ByT5 cross-attention
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-4)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, lengths=None):
        x = self.lit_adapter(x)
        for layer in self.layers:
            if lengths is not None:
                x, lengths = layer(x, lengths)
            else:
                x = layer(x)
        x = self.proj(x)
        return (x, lengths) if lengths is not None else x


class RARIPAMAdapter(RARIPACombined):
    def __init__(
        self,
        num_layers=2,
        ffn_dim=1024,
        kernel=KERNEL,
        stride=STRIDE,
        padding=PADDING,
        dropout=0.1,
        use_mpsa=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.audio_proj = MAdapter(
            in_dim=1024,
            out_dim=self.config.d_model,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dropout=dropout,
            use_mpsa=use_mpsa,
        )

    def _get_projected(self, audio_values, attention_mask=None):
        if attention_mask is None:
            attention_mask = (audio_values != 0).long()

        encoder_out = self.encoder(
            audio_values, attention_mask=attention_mask
        ).last_hidden_state

        feat_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        )

        projected, lengths = self.audio_proj(encoder_out, feat_lengths)

        positions = torch.arange(projected.size(1), device=lengths.device)
        mask = (positions[None, :] < lengths[:, None]).long()

        return projected, mask, lengths, encoder_out, feat_lengths
