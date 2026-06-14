import os
import random
import numpy as np
import torch

# ------------------------------------------------------------------
# AMD / ROCm settings
# ------------------------------------------------------------------
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"

# ------------------------------------------------------------------
# Unsloth imports
# ------------------------------------------------------------------
import unsloth.models._utils as _unsloth_utils

# Prevent HF statistics timeout
_unsloth_utils._get_statistics = lambda *args, **kwargs: None
_unsloth_utils.get_statistics = lambda *args, **kwargs: None

from unsloth import FastLanguageModel

# ------------------------------------------------------------------
# Training imports
# ------------------------------------------------------------------
from transformers import TrainingArguments
from trl import SFTTrainer

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from datasets import load_from_disk
from datasets import DatasetDict

# Load your freshly scrubbed local dataset
train_split = load_from_disk("./telecom_chat_dataset")

# Reconstruct a DatasetDict format so the rest of your script runs unmodified

formatted_ds = DatasetDict({"train": train_split})
print(f"Formatted QnA dataset loaded: {formatted_ds}")

# ------------------------------------------------------------------
# Load Qwen
# ------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=512,
    load_in_4bit=True,
)

# ------------------------------------------------------------------
# LoRA
# ------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ------------------------------------------------------------------
# Convert messages -> training text
# formatted_ds should already look like:
#
# {
#   "messages": [
#       {"role":"user","content":"..."},
#       {"role":"assistant","content":"..."}
#   ]
# }
# ------------------------------------------------------------------
def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

train_dataset = formatted_ds["train"].map(
    format_chat,
    remove_columns=["messages"],
)

print(train_dataset[0]["text"])

# ------------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./telco-qwen",

    num_train_epochs=2,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    weight_decay=0.01,

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    save_strategy="no",
    report_to="none",

    bf16=torch.cuda.is_available(),
    fp16=False,
)

# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,

    dataset_text_field="text",

    max_seq_length=512,

    # Huge speedup for short QA samples
    packing=True,

    args=training_args,
)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
trainer.train()

# ------------------------------------------------------------------
# Save adapter + tokenizer
# ------------------------------------------------------------------
model.save_pretrained("./telco-qwen/final")
tokenizer.save_pretrained("./telco-qwen/final")

print("Training complete.")