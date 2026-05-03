import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

print("Loading datasets...")
train_df = pd.read_csv('data/drugsComTrain_raw.csv')
test_df  = pd.read_csv('data/drugsComTest_raw.csv')

print(f"Training rows: {train_df.shape[0]}")
print(f"Testing rows:  {test_df.shape[0]}")

train_df = train_df[['condition', 'review', 'rating']]
test_df  = test_df[['condition', 'review', 'rating']]

train_df = train_df.dropna()
test_df  = test_df.dropna()
print(f"\nAfter cleaning — Train: {train_df.shape[0]} | Test: {test_df.shape[0]}")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("\nCleaning training reviews... (takes 1-2 mins)")
train_df['clean_review'] = train_df['review'].apply(clean_text)

print("Cleaning testing reviews...")
test_df['clean_review']  = test_df['review'].apply(clean_text)

train_df['input_text'] = train_df['condition'] + ' ' + train_df['clean_review']
test_df['input_text']  = test_df['condition']  + ' ' + test_df['clean_review']

train_df['label'] = (train_df['rating'] >= 6).astype(int)
test_df['label']  = (test_df['rating']  >= 6).astype(int)

print("\nTraining Label Distribution:")
print(train_df['label'].value_counts())
print("\nTesting Label Distribution:")
print(test_df['label'].value_counts())

print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df['input_text']).toarray()
X_test  = vectorizer.transform(test_df['input_text']).toarray()

y_train = train_df['label'].values
y_test  = test_df['label'].values

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Done! All files saved.")