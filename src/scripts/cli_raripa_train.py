import subprocess
import sys
from pathlib import Path

import questionary

from src.utils.cli_utils import get_checkpoints, get_manifest_files

checkpoints = get_checkpoints()
available_manifests = get_manifest_files()

# Prompt user to select the test manifest
test_manifest = questionary.select(
    "Select test data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not test_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select the train manifest
train_manifest = questionary.select(
    "Select train data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not train_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select the validation manifest
valid_manifest = questionary.select(
    "Select validation data manifest:", choices=available_manifests, use_indicator=True
).ask()

if not valid_manifest:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select the validation manifest
mode = questionary.select(
    "Select training mode:",
    choices=["ce-only", "ctc-only", "combined", "m-adapter"],
    use_indicator=True,
).ask()

if not mode:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select a base CTC model to train from
checkpoints.append("Train from scratch")

checkpoint = questionary.select(
    "Select base checkpoint to train from:",
    choices=checkpoints,
    use_indicator=True,
).ask()

description = questionary.text("Enter run description").ask()

# Build the command execution array for uv
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.raripa_train",
    "--test-manifest-path",
    test_manifest,
    "--train-manifest-path",
    train_manifest,
    "--valid-manifest-path",
    valid_manifest,
    "--mode",
    mode,
    "--description",
    description,
]

if checkpoint != "Train from scratch":
    cmd.extend(["--ctc-checkpoint-path", checkpoint])

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
