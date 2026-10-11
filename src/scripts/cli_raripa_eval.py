import subprocess
import sys

import questionary

from src.utils.cli_utils import get_checkpoints, get_manifest_files

# List all available checkpoints
checkpoints = get_checkpoints()

if not checkpoints:
    print("No checkpoints found.")
    sys.exit(1)

# Prompt user to select a checkpoint
selected_checkpoint = questionary.select(
    "Choose a checkpoint to evaluate:", choices=checkpoints, use_indicator=True
).ask()

if not selected_checkpoint:
    print("Cancelled by user.")
    sys.exit(0)

available_manifests = get_manifest_files()

# Prompt user to select a test set
selected_dataset = questionary.select(
    "Choose a testing dataset:", choices=available_manifests, use_indicator=True
).ask()

if not selected_dataset:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select a mode
selected_mode = questionary.select(
    "Choose a mode:", choices=["ce-only", "ctc-only", "combined"], use_indicator=True
).ask()

if not selected_mode:
    print("Cancelled by user.")
    sys.exit(0)

# NOTE: We run the Python script with uv!
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.raripa_eval",
    "--checkpoint-path",
    selected_checkpoint,
    "--test-manifest-path",
    selected_dataset,
    "--mode",
    selected_mode,
]

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
