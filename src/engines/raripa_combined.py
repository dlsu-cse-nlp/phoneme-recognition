import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from src.engines.raripa import RARIPA


class RARIPACombined(RARIPA):
    def __init__(self):
        super().__init__()

        # CTC head projecting to ByT5 vocabulary
        # zero_infinity=True works as a safety net for when characters > frames
        self.ctc_head = nn.Linear(1024, self.config.vocab_size + 1)
        self.ctc_loss = nn.CTCLoss(blank=self.config.vocab_size, zero_infinity=True)
        self.ctc_weight = 0.3

    def forward(
        self,
        audio_values,
        labels=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # Grab the projected inputs to ByT5 as well as the downsampled mask
        projected, mask, projected_lengths, encoder_out, feat_lengths = (
            self._get_projected(audio_values, attention_mask)
        )

        outputs = self.byt5(
            encoder_outputs=(projected,),
            attention_mask=mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        if labels is not None:
            ctc_logits = self.ctc_head(encoder_out)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

            target_lengths = (labels != -100).sum(dim=-1)
            ctc_targets = labels.clone()
            ctc_targets[ctc_targets == -100] = self.config.pad_token_id

            ctc_loss = self.ctc_loss(
                log_probs,
                ctc_targets,
                input_lengths=feat_lengths,
                target_lengths=target_lengths,
            )

            ce_loss = outputs.loss
            # print(f"CE: {ce_loss}; CTC: {ctc_loss}")

            outputs.loss = (self.ctc_weight * ctc_loss) + (
                (1 - self.ctc_weight) * ce_loss
            )

        return outputs

    def generate(self, audio_values, attention_mask=None, ctc_weight=0.3, **kwargs):
        projected, mask, projected_lengths, encoder_out, feat_lengths = (
            self._get_projected(audio_values, attention_mask)
        )
        batch_size = audio_values.size(0)

        num_beams = kwargs.get("num_beams", self.byt5.generation_config.num_beams)
        num_return = kwargs.pop("num_return_sequences", num_beams)

        gen_out = self.byt5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=mask,
            num_return_sequences=num_return,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        sequences = gen_out.sequences  # (B*num_return, L)
        attn_scores = gen_out.sequences_scores  # length-normalized avg log-prob

        # CTC forward pass over the shared encoder output
        ctc_logits = self.ctc_head(encoder_out)  # (B, T, V+1)
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, V+1)
        log_probs = log_probs.repeat_interleave(
            num_return, dim=1
        )  # (T, B*num_return, V+1)
        input_lengths = feat_lengths.repeat_interleave(num_return, dim=0)

        # Format targets
        targets = sequences
        if targets.size(1) > 0:
            starts_with_pad = targets[:, 0] == self.config.pad_token_id
            targets = targets[:, 1:]

        target_lengths = (targets != self.config.pad_token_id).sum(dim=-1).clamp(min=1)

        per_hyp_ctc_nll = F.ctc_loss(
            log_probs,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.config.vocab_size,
            zero_infinity=True,
            reduction="none",
        )  # (B*num_return,) lower = better

        length_penalty = kwargs.get("length_penalty")
        if length_penalty is None:
            length_penalty = getattr(
                self.byt5.generation_config, "length_penalty", None
            )
        if length_penalty is None:
            length_penalty = 1.0

        lengths_pow = target_lengths.float() ** length_penalty

        unnormalized_attn_log_probs = attn_scores * lengths_pow

        ctc_log_probs = -per_hyp_ctc_nll

        combined_unnormalized = (
            1 - ctc_weight
        ) * unnormalized_attn_log_probs + ctc_weight * ctc_log_probs

        # TODO: Normalize by length?
        combined_scores = combined_unnormalized

        combined_scores = combined_scores.view(batch_size, num_return)
        best_idx = combined_scores.argmax(dim=-1)

        sequences = sequences.view(batch_size, num_return, -1)
        return sequences[torch.arange(batch_size), best_idx]
