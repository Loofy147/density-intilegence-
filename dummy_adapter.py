import json
import os
import torch

adapter_config = {
  "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 32,
  "lora_alpha": 16,
  "target_modules": ".*\\.(in_proj|out_proj|up_proj|down_proj)$",
  "lora_dropout": 0.05,
  "bias": "none"
}

os.makedirs("submission", exist_ok=True)
with open("submission/adapter_config.json", "w") as f:
    json.dump(adapter_config, f, indent=2)

# Create a minimal dummy adapter_model.bin
# We need to use valid layer names for a 30B Nemotron model if possible,
# but often vLLM just needs the file to exist and match the config.
# Based on the demo, it uses target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$"
dummy_weights = {
    "base_model.model.layers.0.mixer.in_proj.lora_A.weight": torch.randn(32, 4096),
    "base_model.model.layers.0.mixer.in_proj.lora_B.weight": torch.randn(4096, 32)
}
torch.save(dummy_weights, "submission/adapter_model.bin")

print("Updated dummy adapter with demo target modules.")
