import pandas as pd
import os

print("PREPROCESSING CFPB COMPLAINTS")
df = pd.read_csv('data/raw_data/rows.csv', low_memory=False)
print(f"Original: {len(df):,}")

text_col = 'Consumer complaint narrative'
category_col = 'Product'

df = df.dropna(subset=[text_col, category_col])
df = df[df[text_col].str.strip() != '']
df = df.drop_duplicates(subset=[text_col])

df = df[[text_col, category_col]]
df.columns = ["Ticket Description", "Ticket Type"]

print(f"Final: {len(df):,}, Categories: {df['Ticket Type'].nunique()}")

os.makedirs('data/preprocess_data', exist_ok=True)
df.to_csv('data/preprocess_data/complaints_cleaned.csv', index=False)
print("✓ Saved!")
