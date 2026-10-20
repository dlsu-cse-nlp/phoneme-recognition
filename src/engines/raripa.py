import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file  # <-- Added to load your checkpoint
from transformers import T5ForConditionalGeneration, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

from src.engines.fusion import CONV_DOWNSAMPLE_FACTOR, CONV_KERNEL_SIZE, AudioProjection

WAV2VEC2_HF = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
G2P_CHECKPOINT = "models/g2p/checkpoint-14500"


class RARIPA(nn.Module):
    """An architecture composed of a wav2vec2-based encoder and a ByT5-based
    decoder, connected by a learnable projection.

    ┌───────────────────┐
    │ ENCODER           │
    │ filipino-wav2vec2 │
    │ l-xls-r-300m      │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ Downsampling +    │
    │ Projection        │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ DECODER           │
    │ ByT5 from G2P     │
    └───────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.main_input_name = "audio_values"

        self.encoder = Wav2Vec2Model.from_pretrained(WAV2VEC2_HF)
        self.audio_proj = AudioProjection()

        # Load T5 and strip the encoder to save VRAM
        self.byt5 = T5ForConditionalGeneration.from_pretrained(G2P_CHECKPOINT)
        del self.byt5.encoder

        # Yoink config from the ByT5 model
        self.config = self.byt5.config
        self.generation_config = self.byt5.generation_config

        # Default generation options
        # NOTE: Can be overridden by kwargs in the calling script
        # The generate() call in engines.raripa_combined should also fall back
        # to these if the calling inference script did not provide kwargs
        self.generation_config.max_length = 500  # 256
        self.generation_config.early_stopping = True
        self.generation_config.no_repeat_ngram_size = 24
        self.generation_config.num_beams = 5
        self.generation_config.length_penalty = 0

    def _get_projected(self, audio_values, attention_mask=None):
        """Processes the inputs through the wav2vec2 encoder and the projection.
        Outputs the projected inputs ready for the ByT5 decoder.
        """

        # TODO: Made this return the encoder outputs for now. Probably better
        # to separate it in practice. This is used in the auxiliary CTC head.

        # Fall back to masking out zeroes in case no attention mask is provided
        if attention_mask is None:
            attention_mask = (audio_values != 0).long()

        encoder_out = self.encoder(
            audio_values, attention_mask=attention_mask
        ).last_hidden_state

        projected = self.audio_proj(encoder_out)

        # The wav2vec2 encoder performs downsampling internally
        # Get the sequence lengths after this downsampling, but BEFORE the
        # downsampling in the bridge
        feat_lengths = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        )

        # Get the sequence length AFTER the downsampling in the bridge
        # Output length L_out is given by:
        # (L_in + 2 * padding - dilation * (kernel size - 1) - 1) / stride + 1
        # NOTE: Refer to PyTorch documentation:
        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv1d.html
        projected_lengths = (
            (feat_lengths + 2 * 1 - 1 * (CONV_KERNEL_SIZE - 1) - 1)
            // CONV_DOWNSAMPLE_FACTOR
        ) + 1

        # Inputs are of shape (B, S, D), so .size(1) gets sequence length
        # Clamp lengths to the actual lengths of projected just in case
        projected_lengths = torch.clamp(projected_lengths, min=0, max=projected.size(1))

        # Build a new attention mask with size (B, S)
        downsampled_mask = torch.zeros(
            (attention_mask.shape[0], projected.size(1)),
            dtype=torch.long,
            device=audio_values.device,
        )

        for i, length in enumerate(projected_lengths):
            downsampled_mask[i, :length] = 1

        return projected, downsampled_mask, projected_lengths, encoder_out, feat_lengths

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
        projected, mask, projected_lengths, _, feat_lengths = self._get_projected(
            audio_values, attention_mask
        )

        # Delegate to the forward() call in the ByT5 model bypassing its encoder
        # entirely, using our projected inputs instead
        outputs = self.byt5(
            encoder_outputs=(projected,),
            attention_mask=mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return outputs

    def generate(self, audio_values, attention_mask=None, **kwargs):
        # Grab the projected inputs to ByT5 as well as the downsampled mask
        projected, mask, _, _, feat_lengths = self._get_projected(
            audio_values, attention_mask
        )

        # Delegate to the forward() call in the ByT5 model bypassing its encoder
        # entirely, using our projected inputs instead. This lets us use HF's
        # optimizations
        # NOTE: This reads from self.generation_config before adding kwargs
        return self.byt5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=mask,
            **kwargs,
        )
