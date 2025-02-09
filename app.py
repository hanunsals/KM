import streamlit as st
import re
import string
import nltk
import emoji
import os
import joblib
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Unduh stopwords jika belum diunduh
nltk.download('stopwords')
nltk.download('punkt')

# Load model dan vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(current_dir, "svm_model.pkl")  
vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")  

svm_model = joblib.load(model_path)  
tfidf_vectorizer = joblib.load(vectorizer_path)

# Fungsi preprocessing
def preprocess_text(kalimat):
    kalimat = re.sub(r'@[\w]+', '', kalimat)  # Hapus mention
    kalimat = re.sub(r"\d+", "", kalimat)  # Hapus angka
    kalimat = kalimat.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    kalimat = kalimat.strip()  # Hapus spasi ekstra
    kalimat = re.sub(r'http\S+', '', kalimat)  # Hapus URL
    kalimat = ''.join(c for c in kalimat if not emoji.is_emoji(c))  # Hapus emoji
    return kalimat.lower()

# Fungsi menangani negasi dengan regex
def handle_negation(text):
    negations = {
        r"\btidak jelek\b": "bagus",
        r"\btidak buruk\b": "baik",
        r"\btidak gagal\b": "sukses",
        r"\btidak bodoh\b": "pintar",
        r"\btidak lambat\b": "cepat",
        r"\btidak mahal\b": "murah",
        r"\btidak buruk\b": "baik",
        r"\btidak sulit\b": "mudah",
        r"\btidak salah\b": "benar",
        r"\btidak lemah\b": "kuat"
    }
    for pattern, replacement in negations.items():
        text = re.sub(pattern, replacement, text)
    return text

# Inisialisasi Streamlit
st.title("Analisis Sentimen Kampus Merdeka")
st.write("Masukkan teks untuk dianalisis:")

# Input teks dari user
user_input = st.text_area("Masukkan teks di sini", "")

if st.button("Analisis Sentimen"):
    if user_input:
        # Preprocessing
        cleaned_text = preprocess_text(user_input)
        cleaned_text = handle_negation(cleaned_text)  # Tangani negasi
        
        # Transformasi TF-IDF
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        
        # Prediksi Sentimen
        prediction = svm_model.predict(text_tfidf)
        
        # Mapping label sentimen hanya 2 kelas
        sentiment_map = {0: "Negatif", 1: "Positif"} 
        sentiment = sentiment_map.get(prediction[0], "Tidak diketahui")

        # Output hasil
        st.write(f"Hasil analisis sentimen: **{sentiment}**")
    else:
        st.write("Silakan masukkan teks untuk dianalisis.")
