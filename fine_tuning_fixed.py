import os
import random
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# HuggingFace authentication
# Set HF_TOKEN in your environment before running, e.g.:
#   export HF_TOKEN="hf_xxxxx"
# If no token is set, this will print a warning but won't crash.
# ---------------------------------------------------------------------------
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged in to HuggingFace.")
else:
    print("WARNING: HF_TOKEN not set. Proceeding without authentication.")
    print("Some gated models (like Llama) won't be accessible.")

# ---------------------------------------------------------------------------
# Monkey-patch the unsloth stats check that causes the TimeoutError.
# This is the actual function that hangs for 120s when HF is unreachable.
# We replace it with a no-op BEFORE importing FastLanguageModel.
# ---------------------------------------------------------------------------
import unsloth.models._utils as _unsloth_utils
_unsloth_utils._get_statistics = lambda *args, **kwargs: None
_unsloth_utils.get_statistics  = lambda *args, **kwargs: None

import unsloth
from unsloth import FastLanguageModel

from datasets import load_dataset, Dataset, ClassLabel
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from collections import Counter

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1024**3:.2f} GB")

# ---------------------------------------------------------------------------
# Load dataset
# NOTE: This dataset only has a "train" split – there is NO "test" split.
#       We must create the validation split ourselves.
# ---------------------------------------------------------------------------
ds = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset")
print(f"Raw dataset: {ds}")

# Encode category as ClassLabel for stratified splitting
categories = sorted(set(ds["train"]["category"]))
class_label = ClassLabel(names=categories)

def encode_category(example):
    example["category"] = class_label.str2int(example["category"])
    return example

ds_encoded = ds["train"].map(encode_category)
ds_encoded = ds_encoded.cast_column("category", class_label)

# Stratified train/val split (90/10)
dataset = ds_encoded.train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column="category",
)
train_ds = dataset["train"]
val_ds   = dataset["test"]
print(f"Train: {len(train_ds)} rows, Val: {len(val_ds)} rows")

# ---------------------------------------------------------------------------
# Deduplicate training set
# The dataset has ~16k duplicate responses – remove exact (instruction, response) pairs.
# ---------------------------------------------------------------------------
train_df = train_ds.to_pandas()
before_count = len(train_df)
train_df = train_df.drop_duplicates(subset=["instruction", "response"])
after_count = len(train_df)
print(f"Deduplication: {before_count} -> {after_count} rows (removed {before_count - after_count})")
train_ds = Dataset.from_pandas(train_df, preserve_index=False)

# ---------------------------------------------------------------------------
# Model loading
# Keep load_in_4bit=True – this is unsloth's optimised path (QLoRA).
# The previous 8-bit attempt was wrong: unsloth doesn't properly support it.
# The timeout error was from get_statistics, which we've monkey-patched above.
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=512,
    load_in_4bit=True,       # unsloth's flagship mode – fast QLoRA
)

# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Format data using ChatML template
# ---------------------------------------------------------------------------
def format_chat(example):
    text = (
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    return {"text": text}

train_formatted = train_ds.map(format_chat)
val_formatted   = val_ds.map(format_chat)

print(f"\nSample formatted text:\n{train_formatted[0]['text']}")

# ---------------------------------------------------------------------------
# Training arguments
# NOTE: `packing` belongs in SFTTrainer, NOT in TrainingArguments.
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./telco-qwen",
    num_train_epochs=2,
    per_device_train_batch_size=32,   # was 2 – 192GB VRAM can handle much more
    gradient_accumulation_steps=2,    # was 8 – effective batch = 64
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="no",               # disable checkpoints – unsloth pickle bug
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,     # not needed with 192GB – disabling is faster
    report_to="none",                 # avoid wandb prompt
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_formatted,
    eval_dataset=val_formatted,
    dataset_text_field="text",
    max_seq_length=512,
    packing=True,              # pack short sequences – belongs here, not in args
    args=training_args,
)

# ---------------------------------------------------------------------------
# Train!
# ---------------------------------------------------------------------------
trainer_output = trainer.train()

# ---------------------------------------------------------------------------
# Save the final model manually (avoids the pickle bug in checkpoints)
# ---------------------------------------------------------------------------
model.save_pretrained("./telco-qwen/final")
tokenizer.save_pretrained("./telco-qwen/final")
print("Training complete. Model saved to ./telco-qwen/final")
