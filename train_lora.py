import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configuration
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" # Base model
TRAIN_FILE = "data/combined_train.csv"
OUTPUT_DIR = "nemotron-lora-adapter"

# Load dataset
df = pd.read_csv(TRAIN_FILE)
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def format_instruction(sample):
    # Mimicking the competition prompt format with a "thought" section
    prompt = sample['prompt']
    answer = sample['answer']

    # We encourage the model to "think" before providing the boxed answer
    full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # In a real training, we might have synthetic chain-of-thought.
    # For now, we'll just put the answer in the box.
    full_text += f"The rule is identified. The answer is \\boxed{{{answer}}}.\n</think>\n\\boxed{{{answer}}}<|im_end|>"
    return {"text": full_text}

dataset = dataset.map(format_instruction)

# LoRA Config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Training Arguments (Optimized for a quick run if GPU were available, but keeping it as a template)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none"
)

# Note: In this environment without GPU, we cannot actually run the training of a 30B model.
# This script serves as the "LoRA adapter preparation" logic.
# For the actual submission, I would need to provide the adapter_config.json and the weights.

print("Training script prepared. LoRA Rank:", lora_config.r)
