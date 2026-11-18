import torch
import torch.nn as nn
import torchaudio

from src.engines.ctc_base import BaseCTCModel

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160


class LogMelSubsampler(nn.Module):
    """Extracts 80-bin log-mel spectrograms and subsamples 4x (100 Hz -> 25 Hz) via 2D Conv."""

    def __init__(self, out_dim: int = 256, n_mels: int = N_MELS):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=n_mels,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(64 * freq_out, out_dim)

    def forward(
        self, audio_values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        if attention_mask is None:
            attention_mask = (audio_values != 0).long()

        # MelSpectrogram runs in float32 for numerical stability under bf16 autocast
        with torch.amp.autocast("cuda", enabled=False):
            mel = self.mel(audio_values.float())  # (B, n_mels, T_mel)
            log_mel = torch.log(mel.clamp(min=1e-6))
            mean = log_mel.mean(dim=-1, keepdim=True)
            std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
            log_mel = (log_mel - mean) / std

        x = log_mel.transpose(1, 2).unsqueeze(1)  # (B, 1, T_mel, n_mels)
        x = self.conv(x.to(self.proj.weight.dtype))  # (B, 64, T_sub, F_sub)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.proj(x)

        mel_lengths = (attention_mask.sum(-1) // HOP_LENGTH) + 1
        sub_lengths = ((mel_lengths + 1) // 2 + 1) // 2
        sub_lengths = sub_lengths.clamp(min=1, max=x.size(1))
        return x, sub_lengths


class CNNRNNCTCBaseline(BaseCTCModel):
    """Non-pretrained CNN-RNN-CTC baseline (Log-Mel + 2D Conv + 4-layer BiLSTM + CTC)."""

    def __init__(
        self,
        conv_dim: int = 256,
        rnn_hidden: int = 320,
        num_layers: int = 4,
        vocab_size: int = 256,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__(
            hidden_size=rnn_hidden * 2,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
        )
        self.subsampler = LogMelSubsampler(out_dim=conv_dim)
        self.rnn = nn.LSTM(
            input_size=conv_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def _encode(self, audio_values, attention_mask=None):
        x, lengths = self.subsampler(audio_values, attention_mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        ctc_logits = self.ctc_head(self.dropout(out))
        return ctc_logits, lengths


class ConformerCTCBaseline(BaseCTCModel):
    """Non-pretrained Conformer-CTC baseline (Log-Mel + 2D Conv + Conformer + CTC)."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 6,
        depthwise_conv_kernel_size: int = 15,
        vocab_size: int = 256,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__(
            hidden_size=d_model,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
        )
        self.subsampler = LogMelSubsampler(out_dim=d_model)
        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

    def _encode(self, audio_values, attention_mask=None):
        x, lengths = self.subsampler(audio_values, attention_mask)
        out, out_lengths = self.conformer(x, lengths)
        out_lengths = out_lengths.clamp(min=1, max=out.size(1))
        ctc_logits = self.ctc_head(self.dropout(out))
        return ctc_logits, out_lengths
