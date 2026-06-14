import pandas as pd
df = pd.read_csv('data/train.csv')
def categorize(prompt):
    if "bit manipulation" in prompt: return "bit_manipulation"
    elif "encryption" in prompt or "decrypt" in prompt: return "text_encryption"
    elif "transformation rules is applied to equations" in prompt: return "equations"
    elif "unit conversion" in prompt: return "unit_conversion"
    else: return "other"
df['category'] = df['prompt'].apply(categorize)
other_df = df[df['category'] == 'other']
for i in range(20):
    print(f"\n--- Other Sample {i} ---")
    print(other_df.iloc[i]['prompt'])
    print("Answer:", other_df.iloc[i]['answer'])
