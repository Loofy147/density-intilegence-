import pandas as pd
import re

df = pd.read_csv('data/train.csv')

def categorize(prompt):
    if "bit manipulation" in prompt:
        return "bit_manipulation"
    elif "encryption" in prompt or "decrypt" in prompt:
        return "text_encryption"
    elif "transformation rules is applied to equations" in prompt:
        return "equations"
    elif "unit conversion" in prompt:
        return "unit_conversion"
    else:
        return "other"

df['category'] = df['prompt'].apply(categorize)
print(df['category'].value_counts())

for cat in df['category'].unique():
    print(f"\n--- {cat} ---")
    sample = df[df['category'] == cat].iloc[0]
    print(sample['prompt'])
    print("Answer:", sample['answer'])
