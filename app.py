import streamlit as st
import pandas as pd
import re
import string
import nltk
import emoji
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Unduh stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load model dan vectorizer
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Fungsi preprocessing
def preprocess_text(kalimat):
    kalimat = re.sub(r'@[\w]+', '', kalimat)  # Hapus mention
    kalimat = re.sub(r"\d+", "", kalimat)  # Hapus angka
    kalimat = kalimat.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    kalimat = kalimat.strip()  # Hapus spasi ekstra
    kalimat = re.sub(r'http\S+', '', kalimat)  # Hapus URL
    kalimat = ''.join(c for c in kalimat if not emoji.is_emoji(c))  # Hapus emoji
    return kalimat.lower()

# Inisialisasi Streamlit
st.title("Analisis Sentimen Kampus Merdeka")
st.write("Masukkan teks untuk dianalisis:")

# Input teks dari user
user_input = st.text_area("Masukkan teks di sini", "")

if st.button("Analisis Sentimen"):
    if user_input:
        cleaned_text = preprocess_text(user_input)
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = svm_model.predict(text_tfidf)
        
        sentiment = "Positif" if prediction[0] == 1 else "Negatif"
        st.write(f"Hasil analisis sentimen: **{sentiment}**")
    else:
        st.write("Silakan masukkan teks untuk dianalisis.")
