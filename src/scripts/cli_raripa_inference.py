import subprocess
import sys
from pathlib import Path

import questionary

from src.utils.cli_utils import get_checkpoints

SINGLE_SAMPLES_DIR = Path("data/single_samples")

# List all available checkpoints
checkpoints = get_checkpoints()

if not checkpoints:
    print("No checkpoints found.")
    sys.exit(1)

selected_checkpoint = questionary.select(
    "Choose a checkpoint to evaluate:", choices=checkpoints, use_indicator=True
).ask()

if not selected_checkpoint:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select a mode
selected_mode = questionary.select(
    "Choose a mode:", choices=["ce-only", "ctc-only", "combined"], use_indicator=True
).ask()

if not selected_mode:
    print("Cancelled by user.")
    sys.exit(0)

# List all available .wav files
wav_files = sorted(SINGLE_SAMPLES_DIR.rglob("*.wav"))

if not wav_files:
    print(f"No .wav files found in {SINGLE_SAMPLES_DIR}.")
    sys.exit(1)

# Prompt the user to select a .wav file
selected_wav = questionary.select(
    "Choose a .wav file:", choices=[str(p) for p in wav_files], use_indicator=True
).ask()

if not selected_wav:
    print("Cancelled by user.")
    sys.exit(0)

cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.raripa_inference",
    "--checkpoint-path",
    selected_checkpoint,
    "--wav-path",
    selected_wav,
    "--mode",
    selected_mode,
]

# Prompt the user
attention = questionary.confirm("Map cross-attention?").ask()

if attention:
    output_path = questionary.text("Enter output path:", default="attention.png").ask()

    if output_path:
        cmd.extend(["--attention-map-output", output_path])
    else:
        print("Cancelled. Inference will run without saving a cross-attention map.")

print(f"\nRunning command: {' '.join(cmd)}\n")
subprocess.run(cmd, check=True)
