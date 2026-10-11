import argparse
import json
from datetime import datetime

import editdistance
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from src.datasets.pr_dataset import PhonemeDataCollator, dataset_from_manifests
from src.engines.raripa_build import build_model
from src.utils.preview_callback import PreviewCallback

# Requires manifest files following NeMo's format
TRAIN_MANIFEST = "data/nexdata/filipino_822/train_manifest.json"
VALID_MANIFEST = "data/nexdata/filipino_822/valid_manifest.json"
TEST_MANIFEST = "data/nexdata/filipino_822/test_manifest.json"

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--train-manifest-path", default=TRAIN_MANIFEST)
parser.add_argument("--valid-manifest-path", default=VALID_MANIFEST)
parser.add_argument("--test-manifest-path", default=TEST_MANIFEST)
parser.add_argument("--ctc-checkpoint-path", default=None)
parser.add_argument("--mode", choices=["ce-only", "ctc-only", "combined", "m-adapter"])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)

if args.ctc_checkpoint_path is not None:
    model = build_model(checkpoint_path=args.ctc_checkpoint_path, mode=args.mode)
else:
    model = build_model(mode=args.mode)

dataset = dataset_from_manifests(
    args.train_manifest_path,
    args.valid_manifest_path,
    args.test_manifest_path,
    tokenizer,
)
data_collator = PhonemeDataCollator(tokenizer)

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")


def compute_metrics(eval_preds):
    """Computes the character error rate (CER) during evaluation"""

    # TODO: Implement PER and PFER (?)

    generated, labels = eval_preds

    if isinstance(generated, tuple):
        generated = generated[0]

    generated = np.where(generated != -100, generated, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    total_errors = sum(editdistance.eval(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)

    return {"cer": total_errors / total_chars if total_chars > 0 else 1.0}


training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/checkpoints/{run_id}_raripa",
    bf16=True,
    optim="adamw_torch",
    # fp16=True,
    # --------------------------------------------
    # Effective batch size: 8 * 2 = 16
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    # --------------------------------------------
    # TODO: Hyperparameter tuning
    # learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=200,  # TODO: Use steps
    max_grad_norm=1.0,
    # --------------------------------------------
    num_train_epochs=60,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    # --------------------------------------------
    predict_with_generate=True if args.mode != "ctc-only" else False,
    # --------------------------------------------
    load_best_model_at_end=True,
    metric_for_best_model="cer" if args.mode != "ctc-only" else "loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

# TODO: Can use grouped parameters to control LR...?
# """
optimizer_grouped_parameters = [
    {  # Bridge layers train with a higher LR (1e-3)
        "params": [
            p for n, p in model.audio_proj.named_parameters() if p.requires_grad
        ],
        "lr": 3e-4,
    },
    {  # Rest of the unfrozen layers train with a lower LR (5e-5)
        "params": [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "audio_proj" not in n
        ],
        "lr": 5e-5,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
# """

trainer = Seq2SeqTrainer(
    optimizers=(optimizer, None),
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    callbacks=(
        [PreviewCallback(dataset["validation"], data_collator, tokenizer)]
        if args.mode != "ctc-only"
        else None
    ),
    compute_metrics=compute_metrics if args.mode != "ctc-only" else None,
)

trainer.train()
