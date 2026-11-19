import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput


class BaseCTCModel(nn.Module):
    """Shared CTC training loss and greedy collapse decoding for standalone PR baselines."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int = 256,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.main_input_name = "audio_values"
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.blank_id = vocab_size

        self.config = PretrainedConfig(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            hidden_size=hidden_size,
        )

        self.generation_config = GenerationConfig(
            max_length=500,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        self.dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(hidden_size, vocab_size + 1)
        # zero_infinity=True guards against edge cases where target_lengths > input_lengths
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def _encode(self, audio_values, attention_mask=None):
        raise NotImplementedError

    def forward(
        self,
        audio_values,
        labels=None,
        attention_mask=None,
        **kwargs,
    ):
        ctc_logits, feat_lengths = self._encode(audio_values, attention_mask)

        loss = None
        if labels is not None:
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            valid_mask = (labels != -100) & (labels != self.eos_token_id)
            target_lengths = valid_mask.sum(dim=-1)
            ctc_targets = labels[valid_mask]

            loss = self.ctc_loss(
                log_probs,
                ctc_targets,
                input_lengths=feat_lengths,
                target_lengths=target_lengths,
            )

        return Seq2SeqLMOutput(loss=loss, logits=ctc_logits)

    @torch.no_grad()
    def generate(self, audio_values=None, attention_mask=None, **kwargs):
        """Greedy CTC decoding: argmax per frame -> collapse consecutive repeats -> strip blanks."""
        ctc_logits, feat_lengths = self._encode(audio_values, attention_mask)
        pred_ids = torch.argmax(ctc_logits, dim=-1)

        decoded_sequences = []
        for i in range(pred_ids.size(0)):
            valid_frames = pred_ids[i, : feat_lengths[i]]
            collapsed = torch.unique_consecutive(valid_frames)
            filtered = collapsed[
                (collapsed != self.blank_id)
                & (collapsed != self.pad_token_id)
                & (collapsed != self.eos_token_id)
            ]
            if filtered.numel() == 0:
                filtered = torch.tensor(
                    [self.pad_token_id], device=pred_ids.device, dtype=torch.long
                )
            else:
                filtered = filtered[: self.generation_config.max_length]
            decoded_sequences.append(filtered)

        return torch.nn.utils.rnn.pad_sequence(
            decoded_sequences, batch_first=True, padding_value=self.pad_token_id
        )
