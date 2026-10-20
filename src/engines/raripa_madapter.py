import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engines.raripa_combined import RARIPACombined

# mAda3 setting from the paper: 3 layers, each with k=3, s=2, p=1 (8x overall)
KERNEL, STRIDE, PADDING = 3, 2, 1


class MAdapterLayer(nn.Module):
    def __init__(self, dim=1024, heads=16, ffn_dim=4096):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pool_q, self.pool_k, self.pool_v, self.pool_x = (
            nn.Conv1d(dim, dim, KERNEL, STRIDE, PADDING) for _ in range(4)
        )
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )

    def forward(self, x):  # (B, L, D) -> (B, L', D)
        pool = lambda conv, t: conv(t.transpose(1, 2)).transpose(1, 2)
        pools = (self.pool_q, self.pool_k, self.pool_v)
        q, k, v = (pool(p, t) for p, t in zip(pools, self.qkv(x).chunk(3, dim=-1)))
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        x = self.norm1(pool(self.pool_x, x) + self.out(h))
        return self.norm2(x + self.ffn(x))


class RARIPAMAdapter(RARIPACombined):
    def __init__(self, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.audio_proj = nn.Sequential(
            *[MAdapterLayer() for _ in range(num_layers)],
            nn.Linear(1024, self.config.d_model),  # D=1024 -> ByT5 d_model
        )

    def _get_projected(self, audio_values, attention_mask=None):
        projected, _, _, encoder_out, feat_lengths = super()._get_projected(
            audio_values, attention_mask
        )

        # Attention mask resampling
        lengths = feat_lengths
        for _ in range(self.num_layers):
            lengths = (lengths + 2 * PADDING - KERNEL) // STRIDE + 1

        lengths = lengths.clamp(max=projected.size(1))

        positions = torch.arange(projected.size(1), device=lengths.device)
        mask = (positions[None, :] < lengths[:, None]).long()

        return projected, mask, lengths, encoder_out, feat_lengths
