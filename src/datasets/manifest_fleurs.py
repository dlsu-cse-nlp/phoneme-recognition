import gc
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.sliding_window import predict_sliding_window_batch

RANDOM_STATE = 339
CHECKPOINT = "models/g2p/checkpoint-14500"
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading G2P Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


def get_wav_duration(path):
    return librosa.get_duration(path=str(path))


def process_split(split_name, data_split, tsv_directory):
    """Processes a specific split (train, test, or valid), running G2P and saving the manifest."""
    print(f"\n--- Processing {split_name} split ---")

    wav_files = data_split["audio_filepath"].tolist()
    texts = data_split["sentence"].astype(str).tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"G2P {split_name}"):
        batch_sentences = texts[i : i + BATCH_SIZE]

        batch_preds = predict_sliding_window_batch(
            sentences=batch_sentences,
            model=model,
            tokenizer=tokenizer,
            window_size=11,
            max_batch_size=32,
        )
        predictions.extend(batch_preds)

    data = pd.DataFrame(
        {
            "audio_filepath": wav_files,
            "text": predictions,
            "duration": [
                get_wav_duration(f)
                for f in tqdm(wav_files, desc=f"Durations {split_name}")
            ],
        }
    )

    data["text"] = (
        data["text"]
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"-(?!ʔ)", "ʔ", regex=True)
        .str.replace("-", "", regex=False)
    )

    output_path = Path(tsv_directory) / f"{split_name}_manifest.json"

    data.to_json(
        output_path,
        orient="records",
        lines=True,
        default_handler=str,
    )
    print(f"Saved {len(data)} entries to {output_path}")


def main():
    tsv_directory = Path("data/fleurs/")

    all_data = []

    # Gather all TSV data and preserve valid original audio paths
    for t_type in ["train", "test", "dev"]:
        df = pd.read_csv(
            tsv_directory / f"{t_type}.tsv",
            sep="\t",
            header=None,
            names=[
                "id",
                "audio",
                "sentence",
                "words_only",
                "words_formatted",
                "sampling_rate",
                "gender",
            ],
        )
        df["audio_filepath"] = [
            str(tsv_directory / "audio" / t_type / filename) for filename in df["audio"]
        ]
        all_data.append(df)

    raw_data = pd.concat(all_data, ignore_index=True)

    # Filter dataset
    initial_count = len(raw_data)
    raw_data = raw_data[~raw_data["sentence"].astype(str).str.contains(r"\d", na=False)]
    print(
        f"Filtered out {initial_count - len(raw_data)} rows containing numbers across all datasets."
    )

    raw_data["sentence"] = (
        raw_data["sentence"].astype(str).str.replace(r"[()]", "", regex=True)
    )

    # Create a grouped 60/30/10 split targeting total audio recordings
    # Count how many audio files belong to each unique sentence
    sentence_counts = raw_data.groupby("sentence").size().reset_index(name="counts")

    # Shuffle the unique sentences
    sentence_counts = sentence_counts.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Calculate a running total of audio files
    sentence_counts["cumulative"] = sentence_counts["counts"].cumsum()
    total_audio_files = len(raw_data)

    # Assign splits based on the running total of audio files hitting 60% and 90%
    def assign_split_by_count(cumulative):
        if cumulative <= total_audio_files * 0.60:
            return "train"
        elif cumulative <= total_audio_files * 0.90:
            return "test"
        else:
            return "valid"

    sentence_counts["new_split"] = sentence_counts["cumulative"].apply(
        assign_split_by_count
    )

    # Map the assignments back to the main DataFrame
    raw_data = raw_data.merge(
        sentence_counts[["sentence", "new_split"]], on="sentence", how="left"
    )

    # Process inferences for each newly created subset
    for split in ["train", "test", "valid"]:
        split_data = raw_data[raw_data["new_split"] == split].copy()
        process_split(split, split_data, tsv_directory)

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
