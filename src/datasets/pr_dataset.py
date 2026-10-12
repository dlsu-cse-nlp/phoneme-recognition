import io
import json

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from datasets import DatasetDict

RANDOM_STATE = 765
TARGET_SR = 16000

# Cache files from the parquet here
_parquet_cache: dict[str, pd.DataFrame] = {}


def _load_fsc(path):
    """Used when loading a parquet file, as is the case with FSC"""
    parquet_file, sample_key = path.removeprefix("fsc://").rsplit("/", 1)

    if parquet_file not in _parquet_cache:
        _parquet_cache[parquet_file] = pd.read_parquet(
            "data/fsc/parquet/" + parquet_file
        )

    idx = int(sample_key.removeprefix("sample_").removesuffix(".wav"))
    audio_bytes = _parquet_cache[parquet_file].loc[idx, "audio"]["bytes"]

    return torchaudio.load(io.BytesIO(audio_bytes))


class PhonemeDataset(Dataset):
    def __init__(self, manifest_path, tokenizer):
        with open(manifest_path) as f:
            self.samples = [json.loads(line) for line in f]

        self.samples = sorted(
            self.samples,
            key=lambda s: s["duration"],
        )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["audio_filepath"]

        # TODO: Added a custom prefix for the FSC dataset, but this is probably
        # really clunky so we should figure out a better way. Works for now.
        load_fn = _load_fsc if path.startswith("fsc://") else torchaudio.load
        waveform, sr = load_fn(path)

        # Downsample to target sample rate
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        # In case of stereo audio, average the left and right channels to
        # produce a mono signal
        waveform = waveform.mean(0)

        # z-score normalization for wav2vec2
        # 1e-7 as epsilon to prevent division by zero
        # TODO: Probably better to NOT do this manually...
        waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)

        return {
            "audio_values": waveform,
            "labels": self.tokenizer(
                sample["text"], return_tensors="pt"
            ).input_ids.squeeze(0),
        }


class PhonemeDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Pad audio
        audio = torch.nn.utils.rnn.pad_sequence(
            [item["audio_values"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )

        attention_mask = torch.zeros_like(audio, dtype=torch.long)
        for i, item in enumerate(batch):
            attention_mask[i, : len(item["audio_values"])] = 1

        # Pad labels
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )

        return {
            "audio_values": audio,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def dataset_from_manifests(train_manifest, val_manifest, test_manifest, tokenizer):
    return DatasetDict(
        {
            "train": (
                PhonemeDataset(train_manifest, tokenizer)
                if train_manifest is not None
                else None
            ),
            "validation": (
                PhonemeDataset(val_manifest, tokenizer)
                if val_manifest is not None
                else None
            ),
            "test": (
                PhonemeDataset(test_manifest, tokenizer)
                if test_manifest is not None
                else None
            ),
        }
    )
