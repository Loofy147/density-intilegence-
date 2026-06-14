import pandas as pd
import re

df = pd.read_csv('data/train.csv')

def categorize(prompt):
    if "bit manipulation" in prompt: return "bit_manipulation"
    elif "encryption" in prompt or "decrypt" in prompt: return "text_encryption"
    elif "transformation rules is applied to equations" in prompt: return "equations"
    elif "unit conversion" in prompt: return "unit_conversion"
    elif "numeral system" in prompt: return "numeral_system"
    elif "gravitational constant" in prompt: return "gravity"
    else: return "other"

df['category'] = df['prompt'].apply(categorize)
print(df['category'].value_counts())

for cat in ["equations", "unit_conversion", "gravity"]:
    print(f"\n=== {cat.upper()} EXAMPLES ===")
    samples = df[df['category'] == cat].head(3)
    for _, row in samples.iterrows():
        print(f"Prompt: {row['prompt']}")
        print(f"Answer: {row['answer']}")
        print("-" * 20)
