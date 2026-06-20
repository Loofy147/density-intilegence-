import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
import os

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
TRAIN_FILE = "data/combined_train.csv"
OUTPUT_DIR = "nemotron-lora-adapter"

def format_instruction(sample):
    prompt = sample['prompt']
    answer = sample['answer']
    full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    full_text += "Let's analyze the provided examples to identify the underlying transformation rule.\n"
    full_text += f"The identified rule consistently maps inputs to the observed outputs. Applying this rule to the target input gives the result.\nThe final answer is \\boxed{{{answer}}}.\n</think>\n\\boxed{{{answer}}}<|im_end|>"
    return {"text": full_text}

def train():
    df = pd.read_csv(TRAIN_FILE)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_instruction)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Some versions of TRL use 'max_seq_length' in TrainingArguments or SFTTrainer directly
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    print(f"Ready to train on {len(dataset)} samples.")
    print(f"Base Model: {MODEL_ID}")
    print(f"LoRA Config: Rank={lora_config.r}, Alpha={lora_config.lora_alpha}")

if __name__ == "__main__":
    train()
