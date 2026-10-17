# TODO: Cleanup
# NOTE: Running this is not strictly necessary since the manifest files are
# already pushed to the repository

"""
This script creates three files, test_manifest.json, train_manifest.json, and
valid_manifest.json, in the data/nexdata/filipino_822 directory. The directory
structure is as follows:
data/
├── nexdata/
│   └── filipino_822/
│       ├── G00001/
│       ├── G00608/
│       ├── test_manifest.json
│       ├── train_manifest.json
│       └── valid_manifest.json
└── samples/
"""

import os
from pathlib import Path

import pandas as pd
import torch
from epitran.backoff import Backoff
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.file_read import get_wav_duration, read_file
from src.utils.train_test_val_split import train_test_val_split

CHECKPOINT = "models/g2p/checkpoint-14500"
BATCH_SIZE = 16
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


def make_manifest(_directory):
    """Creates a manifest file for the NexData 822-hour Filipino speech corpus"""
    directory = Path(_directory)

    # Old version using Epitran
    # epi = epitran.Epitran("tgl-Latn")
    # epi = Backoff(["tgl-Latn", "eng-Latn"])

    wav_files = [
        directory / file for file in os.listdir(directory) if file.endswith(".wav")
    ]
    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]

    texts = [read_file(directory / filename).strip() for filename in sorted(txt_files)]

    # Batch inference with progress bar
    predictions = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"G2P {directory.name}"):
        batch = texts[i : i + BATCH_SIZE]
        predictions.extend(predict_batch(batch))

    # Create the needed data for the manifest file
    data = pd.DataFrame(
        {
            "audio_filepath": sorted(wav_files),
            "text": predictions,
            "duration": [get_wav_duration(filename) for filename in sorted(wav_files)],
        }
    )
    print(data)
    return data


data = pd.concat(
    [
        make_manifest("data/nexdata/audio/G00001"),
        make_manifest("data/nexdata/audio/G00608"),
    ]
)
train_test_val_split(data, "data/nexdata/filipino_822/")
