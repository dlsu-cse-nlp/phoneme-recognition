from src.engines.mdd_build import build_mdd_model, save_mdd_model

KOTONE_CHECKPOINT = "models/checkpoints/2026-07-02_17-52_kotone/checkpoint-2340"
G2P_CHECKPOINT = "models/g2p/checkpoint-14500"
TOKENIZER_SOURCE = "models/g2p/checkpoint-14500"
SAVE_DIR = "models/checkpoints/mdd_model_initialized"

if __name__ == "__main__":
    model, tokenizer = build_mdd_model(
        KOTONE_CHECKPOINT, G2P_CHECKPOINT, TOKENIZER_SOURCE, device="cpu"
    )
    save_mdd_model(model, tokenizer, SAVE_DIR)
