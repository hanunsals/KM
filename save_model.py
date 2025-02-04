import joblib

# Simpan model yang sudah dilatih
joblib.dump(svm_model, "svm_model.pkl")

# Simpan TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("Model dan vectorizer berhasil disimpan!")
