import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer

def create_submission_zip(adapter_path, output_zip="submission.zip"):
    import shutil
    if os.path.exists("submission_temp"):
        shutil.rmtree("submission_temp")
    os.makedirs("submission_temp")

    # Copy adapter files
    for f in ["adapter_config.json", "adapter_model.bin"]:
        src = os.path.join(adapter_path, f)
        if os.path.exists(src):
            shutil.copy(src, "submission_temp")
        else:
            print(f"Warning: {f} not found in {adapter_path}")

    # Create the zip
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', "submission_temp")
    print(f"Created {output_zip}")

if __name__ == "__main__":
    # This would be called after training
    # create_submission_zip("nemotron-lora-adapter")
    pass
