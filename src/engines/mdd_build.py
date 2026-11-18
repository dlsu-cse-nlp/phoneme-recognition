from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
)

from src.engines.mdd import MDDModel
from src.engines.raripa import WAV2VEC2_HF
from src.engines.raripa_build import build_model as build_kotone


def build_mdd_model(kotone_checkpoint, g2p_checkpoint, tokenizer_source, device="cpu"):
    # Assembles MDD model from RARIPA checkpoint and ByT5 G2P model checkpoint.
    kotone = build_kotone(checkpoint_path=kotone_checkpoint, mode="combined")

    # Load the ByT5 G2P Seq2Seq model
    g2p_model = AutoModelForSeq2SeqLM.from_pretrained(g2p_checkpoint)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Pass both models to MDDModel
    model = MDDModel(kotone, g2p_model).to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded MDD model with {total:,} parameters -- all pretrained, "
        "nothing new to train."
    )

    return model, tokenizer


def _build_skeleton(g2p_config_source, device="cpu"):
    # Builds an MDDModel
    kotone = build_kotone(checkpoint_path=None, mode="combined")

    g2p_config = AutoConfig.from_pretrained(str(g2p_config_source))
    g2p_config.tie_word_embeddings = False
    g2p_model = AutoModelForSeq2SeqLM.from_config(g2p_config)

    return MDDModel(kotone, g2p_model).to(device)


def save_mdd_model(model, tokenizer, save_dir):
    # One-time snapshot of an already-assembled MDDModel's weights.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "mdd_model_state_dict.pt")
    tokenizer.save_pretrained(save_dir)
    Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_HF).save_pretrained(save_dir)
    model.g2p_model.config.save_pretrained(save_dir / "g2p_config")

    print(f"Saved MDD model state_dict + tokenizer + feature extractor to {save_dir}")


def load_mdd_model(save_dir, device="cpu"):
    # Loads snapshot written by save_mdd_model()
    save_dir = Path(save_dir)

    model = _build_skeleton(save_dir / "g2p_config", device=device)

    state_dict = torch.load(save_dir / "mdd_model_state_dict.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(save_dir)

    return model, tokenizer, feature_extractor
