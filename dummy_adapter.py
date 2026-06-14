import json
import os

adapter_config = {
  "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 32,
  "lora_alpha": 64,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "lora_dropout": 0.05,
  "bias": "none"
}

os.makedirs("submission", exist_ok=True)
with open("submission/adapter_config.json", "w") as f:
    json.dump(adapter_config, f, indent=2)

# Create a dummy adapter_model.bin (normally this would be the trained weights)
# In a real scenario, this would be saved by model.save_pretrained()
import torch
dummy_weights = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((32, 4096))}
torch.save(dummy_weights, "submission/adapter_model.bin")

print("Dummy adapter structure created in 'submission' directory.")
