from safetensors.torch import load_file

from src.engines.raripa import RARIPA
from src.engines.raripa_combined import RARIPACombined
from src.engines.raripa_ctc import RARIPACTC
from src.engines.raripa_madapter import RARIPAMAdapter
from src.engines.rrl_baselines import CNNRNNCTCBaseline, ConformerCTCBaseline
from src.engines.wav2vec2_ctc import (
    HUBERT_HF,
    WAV2VEC2_HF,
    XLSR_HF,
    Wav2Vec2CTCBaseline,
)


def build_model(
    checkpoint_path=None,
    mode="ce-only",
    madapter_layers=2,
    madapter_ffn_dim=1024,
):
    if mode == "ctc-only":
        model = RARIPACTC()
    elif mode == "wav2vec2-ctc":
        model = Wav2Vec2CTCBaseline(encoder_id=WAV2VEC2_HF)
    elif mode == "xlsr-ctc":
        model = Wav2Vec2CTCBaseline(encoder_id=XLSR_HF)
    elif mode == "hubert-ctc":
        model = Wav2Vec2CTCBaseline(encoder_id=HUBERT_HF)
    elif mode == "cnn-rnn-ctc":
        model = CNNRNNCTCBaseline()
    elif mode == "conformer-ctc":
        model = ConformerCTCBaseline()
    elif mode == "combined":
        model = RARIPACombined()
    elif mode in ("m-adapter", "madapter"):
        model = RARIPAMAdapter(
            num_layers=madapter_layers,
            ffn_dim=madapter_ffn_dim,
        )
    else:
        model = RARIPA()

    if checkpoint_path is not None:
        state_dict = load_file(checkpoint_path + "/model.safetensors")
        model.load_state_dict(state_dict, strict=True)

    # TODO: Investigate max length and beam search more

    # TODO: Investigate selective freezing more

    # Freeze pretrained speech encoder except for top 6 layers
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False

        for layer in model.encoder.encoder.layers[-6:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Freeze ByT5 decoder except for cross-attention and self-attention
    if hasattr(model, "byt5"):
        for param in model.byt5.parameters():
            param.requires_grad = False

        for name, param in model.byt5.decoder.named_parameters():
            if "EncDecAttention" in name or "SelfAttention" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    if hasattr(model, "ctc_head"):
        for param in model.ctc_head.parameters():
            param.requires_grad = True

    # Print number of trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    return model
