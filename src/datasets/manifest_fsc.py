# TODO: Cleanup
# NOTE: Running this is not strictly necessary since the manifest files are
# already pushed to the repository

"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/filipinospeechcorpus/data directory.
"""

import io
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

CHECKPOINT = "models/g2p/checkpoint-14500"
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


def predict_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def process_parquet_dataset(directory_path):
    """Reads parquet files, skips audio extraction, runs G2P, and aggregates data."""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"The directory {directory.resolve()} does not exist!")

    parquet_files = [
        directory / f for f in os.listdir(directory) if f.endswith(".parquet")
    ]
    print(f"Found {len(parquet_files)} parquet files.")

    # Fix 1: separate files by prefix instead of pooling all together
    train_files = [f for f in parquet_files if f.name.startswith("train")]
    test_files = [f for f in parquet_files if f.name.startswith("test")]

    def process_files(files):
        all_audio_paths, all_predictions, all_durations = [], [], []

        # Fix 2: load all files in the split at once so G2P runs over larger batches
        frames = []
        for parquet_file in files:
            df = pd.read_parquet(parquet_file)
            df["_source"] = parquet_file.name
            df["_local_idx"] = range(len(df))
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["audio_filepath", "text", "duration"])

        combined = pd.concat(frames, ignore_index=True)

        all_audio_paths = [
            f"data/fsc/parquet/{row['_source']}/sample_{int(row['_local_idx']):06d}.wav"
            for _, row in combined.iterrows()
        ]
        all_durations = combined["duration"].tolist()

        texts = (
            combined["sentence"]
            .astype(str)
            .str.strip()
            .apply(
                lambda s: (
                    s if s.endswith((".", "!", "?")) or len(s.split()) == 1 else s + "."
                )
            )
            .str.capitalize()
            .tolist()
        )

        with tqdm(total=len(texts), desc="G2P") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                all_predictions.extend(predict_batch(batch))
                pbar.update(len(batch))

        data = pd.DataFrame(
            {
                "audio_filepath": all_audio_paths,
                "text": all_predictions,
                "duration": all_durations,
            }
        )
        data["text"] = (
            data["text"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
            .str.replace("-", "", regex=False)
        )
        return data

    train_data = process_files(train_files)
    test_data = process_files(test_files)
    return train_data, test_data


BASE_DIR = "data/fsc/parquet"
train_data, test_data = process_parquet_dataset(BASE_DIR)

if not train_data.empty:
    print("\nSplitting train dataset 90/10 into train/dev...")
    train_df, dev_df = train_test_split(train_data, test_size=0.10, random_state=42)

    output_path = Path("data/fsc")
    train_df.to_json(output_path / "train_manifest.json", orient="records", lines=True)
    dev_df.to_json(output_path / "valid_manifest.json", orient="records", lines=True)

if not test_data.empty:
    test_data.to_json(output_path / "test_manifest.json", orient="records", lines=True)
