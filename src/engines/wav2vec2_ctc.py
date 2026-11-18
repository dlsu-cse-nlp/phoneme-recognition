import torch
from transformers import AutoModel

from src.engines.ctc_base import BaseCTCModel

WAV2VEC2_HF = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
XLSR_HF = "facebook/wav2vec2-xls-r-300m"
HUBERT_HF = "facebook/hubert-large-ll60k"


class Wav2Vec2CTCBaseline(BaseCTCModel):
    """Standalone self-supervised speech encoder (Wav2Vec 2.0 / XLS-R / HuBERT) + Linear CTC."""

    def __init__(
        self,
        encoder_id: str = WAV2VEC2_HF,
        vocab_size: int = 256,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        dropout: float = 0.1,
    ):
        encoder = AutoModel.from_pretrained(encoder_id)
        super().__init__(
            hidden_size=encoder.config.hidden_size,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
        )
        self.encoder = encoder
        self.config = self.encoder.config
        self.config.pad_token_id = pad_token_id
        self.config.eos_token_id = eos_token_id

    def _encode(self, audio_values, attention_mask=None):
        if attention_mask is None:
            attention_mask = (audio_values != 0).long()

        encoder_out = self.encoder(
            audio_values, attention_mask=attention_mask
        ).last_hidden_state

        feat_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        )
        feat_lengths = torch.clamp(feat_lengths, min=1, max=encoder_out.size(1))

        ctc_logits = self.ctc_head(self.dropout(encoder_out))
        return ctc_logits, feat_lengths
