import argparse

import matplotlib.pyplot as plt
import torch
import torchaudio
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from src.engines.raripa_build import build_model

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
TARGET_SR = 16000

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--wav-path", required=True)
parser.add_argument(
    "--mode", required=True, choices=["ce-only", "ctc-only", "combined", "m-adapter"]
)
parser.add_argument(
    "--layer",
    type=int,
    default=-1,
    help="Decoder layer for the attention map (-1 averages over all layers)",
)
parser.add_argument(
    "--attention-map-output",
    default=None,
    help="If given, also saves a cross-attention heatmap to this path",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)

model = build_model(mode=args.mode)
model.load_state_dict(
    load_file(f"{args.checkpoint_path}/model.safetensors"), strict=False
)
model.to(device)
model.eval()

# TODO: Can import from PRDataset or something
waveform, sr = torchaudio.load(args.wav_path)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
waveform = waveform.mean(0)
waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
audio = waveform.unsqueeze(0).to(device)  # (1, T)
attention_mask = torch.ones_like(audio, dtype=torch.long)

with torch.no_grad():
    ids = model.generate(audio, attention_mask)
    print(
        "-" * 40,
        "\ngenerate() output:",
        tokenizer.decode(ids[0], skip_special_tokens=True),
        "\n",
        "-" * 40,
    )

    if args.attention_map_output is not None:
        # With beams, the attentions would be mixed across hypotheses
        projected, mask, *_ = model._get_projected(audio, attention_mask)
        outputs = model.byt5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=mask,
            num_beams=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )

if args.attention_map_output is not None:
    generated_ids = outputs.sequences[0]
    print(
        "Greedy decode used for attention plot:",
        tokenizer.decode(generated_ids, skip_special_tokens=True),
    )

    attn = torch.stack([torch.stack(step) for step in outputs.cross_attentions])
    attn = attn[:, :, 0, :, -1]  # (steps, layers, heads, src_len)
    if args.layer == -1:
        attn_matrix = attn.mean(dim=(1, 2))  # Average over layers and heads
    else:
        attn_matrix = attn[:, args.layer].mean(dim=1)  # Average over heads
    attn_matrix = attn_matrix.cpu()  # (steps, src_len)

    num_steps = attn_matrix.size(0)
    offset = getattr(tokenizer, "offset", 3)

    token_labels = []
    for token_id in generated_ids[1 : 1 + num_steps].tolist():
        byte_val = token_id - offset
        if 0 <= byte_val <= 255:
            char = bytes([byte_val]).decode("utf-8", errors="replace")
            token_labels.append(f"0x{byte_val:02X} {char!r}")
        else:
            token_labels.append(f"[{tokenizer.convert_ids_to_tokens(token_id)}]")

    fig, ax = plt.subplots(figsize=(11, max(4, 0.3 * num_steps)))
    im = ax.imshow(attn_matrix.numpy(), aspect="auto", cmap="viridis")

    ax.set_xlabel("Encoder frame")
    ax.set_ylabel("Decoder step (predicted token)")
    ax.set_yticks(range(num_steps))
    ax.set_yticklabels(token_labels)
    ax.set_title("Cross-attention: decoder step vs encoder frame")

    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(args.attention_map_output, dpi=150)

    print(f"Saved attention heatmap to {args.attention_map_output}")
