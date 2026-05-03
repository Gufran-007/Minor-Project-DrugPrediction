import pandas as pd

df = pd.read_csv('data/drugsComTrain_raw.csv')

print("=" * 50)
print("DATASET SIZE:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n" + "=" * 50)
print("COLUMN NAMES:")
print(df.columns.tolist())

print("\n" + "=" * 50)
print("FIRST 5 ROWS OF DATA:")
print(df.head())

print("\n" + "=" * 50)
print("MISSING VALUES IN EACH COLUMN:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("RATING DISTRIBUTION (how many reviews per rating):")
print(df['rating'].value_counts().sort_index())

print("\n" + "=" * 50)
print("EXAMPLE REVIEW (row 0):")
print(f"Drug:      {df['drugName'].iloc[0]}")
print(f"Condition: {df['condition'].iloc[0]}")
print(f"Review:    {df['review'].iloc[0]}")
print(f"Rating:    {df['rating'].iloc[0]}")
