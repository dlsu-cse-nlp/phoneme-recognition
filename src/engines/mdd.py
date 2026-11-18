import math

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from src.engines.fusion import CONV_DOWNSAMPLE_FACTOR
from src.engines.raripa_combined import RARIPACombined
from src.utils.mdd_diagnosis import align_phonemes, unit_index_per_byte


class MDDModel(nn.Module):
    # MDD model from RARIPA checkpoint and ByT5 G2P model checkpoint.

    def __init__(self, kotone: RARIPACombined, g2p_model):
        super().__init__()
        self.kotone = kotone
        self.g2p_model = g2p_model
        for param in self.kotone.parameters():
            param.requires_grad = False
        for param in self.g2p_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def predict_canonical_pronunciation(self, tokenizer, expected_text):
        # Runs the G2P model over grapheme input text to predict canonical phoneme sequence
        device = next(self.g2p_model.parameters()).device
        inputs = tokenizer(expected_text, return_tensors="pt").to(device)
        generated_ids = self.g2p_model.generate(
            **inputs, max_length=256, num_beams=4, early_stopping=True
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def analyze(
        self,
        audio_values,
        tokenizer,
        expected_text,
        audio_attention_mask=None,
        delimiter=" ",
        sampling_rate=16000,
    ):
        """
        audio_values: pre-extracted wav2vec2 input (see Wav2Vec2FeatureExtractor
            in test_mdd.py), shape (1, samples). Batch size 1 only.
        tokenizer: the ByT5 tokenizer shared by KoToNe's byt5 decoder and
            the G2P model (both were initialized from the same checkpoint).
        expected_text: the GRAPHEME text the speaker was asked to say (e.g.
            "mangagamot"), NOT phonemes -- the canonical pronunciation is
            predicted from this by the G2P model.
        delimiter: separator between phoneme units in the model's own
            (predicted and recognized) phoneme strings. " " for
            space-separated phonemes; "" for a plain character-per-phoneme
            string (note: multi-byte IPA symbols lose some precision in ""
            mode -- see `unit_index_per_byte`).
        sampling_rate: sampling rate audio_values was extracted at (must
            match the Wav2Vec2FeatureExtractor used upstream). Only used to
            convert frame indices into seconds for the TextGrid export.
        """
        expected_pronunciation = self.predict_canonical_pronunciation(
            tokenizer, expected_text
        )

        if audio_attention_mask is None:
            audio_attention_mask = (audio_values != 0).long()

        projected, frame_mask, frame_lengths, _, _ = self.kotone._get_projected(
            audio_values, audio_attention_mask
        )
        n_frames = int(frame_lengths[0])

        # Recognize what was actually said, using ByT5 decoder
        raw_generated = self.kotone.byt5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=frame_mask,
        )[0]

        eos_id = tokenizer.eos_token_id
        ids = raw_generated.tolist()[1:]  # drop the leading decoder-start token
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        content_ids = ids  # recognized phoneme content, as byte-level token ids

        recognized_text = tokenizer.decode(content_ids, skip_special_tokens=True)

        # Re-run the SAME trained decoder with those ids as input (teacher
        # forcing) purely to read off its cross-attention.
        labels = torch.tensor([content_ids + [eos_id]], device=audio_values.device)
        out = self.kotone.byt5(
            encoder_outputs=(projected,),
            attention_mask=frame_mask,
            labels=labels,
            output_attentions=True,
        )
        cross_attn = torch.stack(out.cross_attentions, dim=0)  # (L, B, H, T_dec, T_enc)
        avg_attn = cross_attn.mean(dim=(0, 2))[0]  # (T_dec, T_enc)
        avg_attn = avg_attn[: len(content_ids)]  # drop the row that predicts EOS

        # For each audio frame, which recognized byte was it "for".
        frame_to_content_pos = avg_attn[:, :n_frames].argmax(dim=0)  # (n_frames,)

        # Align the G2P-predicted canonical form against the recognized pronunciation.
        events = align_phonemes(expected_pronunciation, recognized_text, delimiter)
        # Events are in recognized-order; keep only ones with a recognized
        # counterpart (drops pure deletions, which touch no audio frame).
        unit_events = [e for e in events if e["actual"] is not None]

        # Map each recognized byte position to a recognized-unit index so we
        # can look up its diagnosis event.
        delimiter_id = (
            tokenizer(delimiter, add_special_tokens=False)["input_ids"][0]
            if delimiter
            else None
        )
        unit_of_byte = unit_index_per_byte(content_ids, delimiter_id)

        frame_reports = []
        for t in range(n_frames):
            pos = int(frame_to_content_pos[t])
            unit_idx = unit_of_byte[pos] if pos < len(unit_of_byte) else None
            if unit_idx is not None and unit_idx < len(unit_events):
                ev = unit_events[unit_idx]
                mispronounced = ev["type"] != "correct"
                expected_unit, actual_unit = ev["expected"], ev["actual"]
            else:
                mispronounced, expected_unit, actual_unit = False, None, None
            frame_reports.append(
                {
                    "frame": t,
                    "expected": expected_unit,
                    "actual": actual_unit,
                    "mispronounced": mispronounced,
                }
            )

        # Timing, for the TextGrid export: wav2vec2's own feature-extractor
        # downsampling, times the adapter's additional stride-2 conv.
        wav2vec2_stride = math.prod(self.kotone.encoder.config.conv_stride)
        frame_duration = (wav2vec2_stride * CONV_DOWNSAMPLE_FACTOR) / sampling_rate

        if audio_attention_mask is not None:
            n_valid_samples = int(audio_attention_mask[0].sum().item())
        else:
            n_valid_samples = audio_values.shape[-1]
        total_duration = n_valid_samples / sampling_rate

        return {
            "expected_pronunciation": expected_pronunciation,
            "recognized_text": recognized_text,
            "events": events,
            "frame_reports": frame_reports,
            "frame_duration": frame_duration,
            "total_duration": total_duration,
        }
