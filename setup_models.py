import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data yang sudah dipreprocess
print("Loading preprocessed data...")
df = pd.read_csv('Clean_dataset (1).csv')

# Pastikan kolom 'cleaned_name' ada
if 'cleaned_name' not in df.columns:
    print("ERROR: Kolom 'cleaned_name' tidak ditemukan!")
    print("Pastikan preprocessing sudah selesai dengan lengkap.")
    exit()

print(f"Data loaded: {len(df)} rows")

# Buat TF-IDF Vectorizer
print("\nCreating TF-IDF model...")
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['cleaned_name'])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Simpan model
print("\nSaving models...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ tfidf_vectorizer.pkl saved")

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
print("✓ tfidf_matrix.pkl saved")

print("\n✅ Setup complete! Dashboard siap dijalankan.")