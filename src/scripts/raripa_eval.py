import argparse
import os
from pathlib import Path

import editdistance
import numpy as np
import pandas as pd
import panphon
import panphon.distance
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.datasets.pr_dataset import PhonemeDataCollator, dataset_from_manifests
from src.engines.raripa_build import build_model
from src.metrics.evaluation import Metrics

TEST_MANIFEST = "data/nexdata/filipino_822/test_manifest.json"
DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
BATCH_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", default=TEST_MANIFEST)
parser.add_argument("--test-manifest-path", default=TEST_MANIFEST)
parser.add_argument(
    "--mode",
    choices=[
        "ce-only",
        "ctc-only",
        "combined",
        "m-adapter",
        "wav2vec2-ctc",
        "xlsr-ctc",
        "hubert-ctc",
        "cnn-rnn-ctc",
        "conformer-ctc",
    ],
)
parser.add_argument("--madapter-layers", type=int, default=2)
parser.add_argument("--madapter-ffn-dim", type=int, default=1024)
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)

model = build_model(
    mode=args.mode,
    madapter_layers=args.madapter_layers,
    madapter_ffn_dim=args.madapter_ffn_dim,
)
model.load_state_dict(
    load_file(f"{args.checkpoint_path}/model.safetensors"), strict=False
)
model.to(device)
model.eval()

dataset = dataset_from_manifests(None, None, args.test_manifest_path, tokenizer)
data_collator = PhonemeDataCollator(tokenizer)
test_set = dataset["test"]

metrics = Metrics()

for i in range(0, len(test_set), BATCH_SIZE):
    batch_list = [
        {k: v for k, v in test_set[j].items() if k in ["audio_values", "labels"]}
        for j in range(i, min(i + BATCH_SIZE, len(test_set)))
    ]

    inputs = data_collator(batch_list)
    audio = inputs["audio_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated = model.generate(audio, attention_mask)

    labels = inputs["labels"].clone()
    labels[labels == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate pooled and per-sample metrics while also print predictions
    for idx, (p, l) in enumerate(zip(preds, refs)):
        item = test_set[i + idx]
        sentence = item.get("sentence", "") if hasattr(item, "get") else ""

        metrics.get_sample_all(p, l, save=True, sentence=sentence)


print(metrics.get_pooled_metrics())

# Convert per-sample results to Pandas, same as the ByT5 eval script
df = metrics.get_output_df()

print("-" * 40)
print("Per-sample error statistics")
print("-" * 40)
print(df[["per", "pfer", "sper"]].describe())

pkl_path = (
    f"results/output_{Path(args.checkpoint_path).parts[-2]}"
    f"_{Path(args.checkpoint_path).parts[-1]}.pkl"
)
df.to_pickle(pkl_path)
print(f"Per-sample results saved to {pkl_path}")
