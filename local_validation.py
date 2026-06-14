import pandas as pd
import re
import numpy as np
import string

def categorize(prompt):
    if "bit manipulation" in prompt: return "bit_manipulation"
    elif "encryption" in prompt or "decrypt" in prompt: return "text_encryption"
    elif "transformation rules is applied to equations" in prompt: return "equations"
    elif "unit conversion" in prompt: return "unit_conversion"
    elif "numeral system" in prompt: return "numeral_system"
    elif "gravitational constant" in prompt: return "gravity"
    else: return "other"

def solve_text_encryption(prompt):
    # Extract pairs: encrypted -> decrypted
    examples_text = re.search(r"Here are some examples:\n(.*?)\nNow, decrypt", prompt, re.DOTALL)
    if not examples_text: return None

    mapping = {}
    lines = examples_text.group(1).strip().split('\n')
    for line in lines:
        if " -> " in line:
            enc, dec = line.split(" -> ")
            for e_char, d_char in zip(enc, dec):
                if e_char.isalpha():
                    mapping[e_char] = d_char

    target_match = re.search(r"decrypt the following text: (.*)", prompt)
    if target_match:
        target = target_match.group(1).strip()
        result = "".join(mapping.get(c, c) for c in target)
        return result
    return None

df = pd.read_csv('data/train.csv')
df['category'] = df['prompt'].apply(categorize)

print("Validating Text Encryption...")
text_df = df[df['category'] == 'text_encryption'].head(100)
correct = 0
for _, row in text_df.iterrows():
    pred = solve_text_encryption(row['prompt'])
    if pred is not None and pred.strip() == row['answer'].strip():
        correct += 1
print(f"Text Encryption Accuracy: {correct}/{len(text_df)}")
