from safetensors.torch import load_file

from src.engines.raripa import RARIPA
from src.engines.raripa_combined import RARIPACombined
from src.engines.raripa_ctc import RARIPACTC


def build_model(checkpoint_path=None, mode="ce-only"):
    if mode == "ctc-only":
        model = RARIPACTC()
    elif mode == "combined":
        model = RARIPACombined()
    else:
        model = RARIPA()

    if checkpoint_path is not None:
        state_dict = load_file(checkpoint_path + "/model.safetensors")
        model.load_state_dict(state_dict, strict=True)

    # TODO: Investigate max length and beam search more

    # TODO: Investigate selective freezing more

    # Freeze wav2vec2 encoder except for top 6 layers
    for param in model.encoder.parameters():
        param.requires_grad = False

    # """
    for layer in model.encoder.encoder.layers[-6:]:
        for param in layer.parameters():
            param.requires_grad = True
    # """

    # Freeze ByT5 decoder except for cross-attention and self-attention
    for param in model.byt5.parameters():
        param.requires_grad = False

    for name, param in model.byt5.decoder.named_parameters():
        if "EncDecAttention" in name or "SelfAttention" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if mode in ["ctc-only", "combined"]:
        for param in model.ctc_head.parameters():
            param.requires_grad = True

    # Print number of trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    return model
