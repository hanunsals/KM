import streamlit as st  # Library untuk membuat aplikasi web interaktif
import re  # Library untuk ekspresi reguler (regex)
import string  # Library untuk manipulasi string
import nltk  # Library untuk pemrosesan bahasa alami
import emoji  # Library untuk menghapus emoji dari teks
import os  # Library untuk berinteraksi dengan sistem file
import joblib  # Library untuk memuat model yang telah disimpan
from nltk.tokenize import word_tokenize  # Fungsi untuk tokenisasi kata
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF Vectorizer
from sklearn.svm import SVC  # Support Vector Machine untuk klasifikasi

# Unduh stopwords jika belum diunduh (diperlukan untuk preprocessing)
nltk.download('stopwords')
nltk.download('punkt')

# Mendapatkan path direktori saat ini untuk memastikan file model bisa diakses
current_dir = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(current_dir, "svm_model.pkl")  # Model ini sudah dilatih sebelumnya untuk mengenali pola sentimen dalam data
vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")  # TF-IDF digunakan karena dapat mengubah teks menjadi representasi numerik yang lebih baik 

# Memuat model SVM dan TF-IDF Vectorizer yang telah dilatih sebelumnya
svm_model = joblib.load(model_path)  
tfidf_vectorizer = joblib.load(vectorizer_path)

# Fungsi preprocessing teks
def preprocess_text(kalimat):
    """
    Membersihkan teks dari karakter yang tidak diperlukan agar model dapat bekerja lebih optimal.
    """
    kalimat = re.sub(r'@[\w]+', '', kalimat)  # Hapus mention (@username)
    kalimat = re.sub(r"\d+", "", kalimat)  # Hapus angka
    kalimat = kalimat.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    kalimat = kalimat.strip()  # Hapus spasi ekstra di awal dan akhir teks
    kalimat = re.sub(r'http\S+', '', kalimat)  # Hapus URL
    kalimat = ''.join(c for c in kalimat if not emoji.is_emoji(c))  # Hapus emoji
    return kalimat.lower()  # Ubah teks menjadi huruf kecil untuk konsistensi

# Fungsi untuk menangani negasi dalam teks
def handle_negation(text):
    """
    Mengubah frasa negasi menjadi kata yang memiliki makna positif.
    Contoh: "tidak jelek" → "bagus".
    """
    negations = {
        r"\btidak jelek\b": "bagus",
        r"\btidak buruk\b": "baik",
        r"\btidak gagal\b": "sukses",
        r"\btidak bodoh\b": "pintar",
        r"\btidak lambat\b": "cepat",
        r"\btidak mahal\b": "murah",
        r"\btidak sulit\b": "mudah",
        r"\btidak salah\b": "benar",
        r"\btidak lemah\b": "kuat"
    }
    for pattern, replacement in negations.items():
        text = re.sub(pattern, replacement, text)  # Mengganti frasa dengan makna positif
    return text

# Fungsi untuk mendeteksi apakah teks hanya berisi angka atau simbol
def is_invalid_input(text):
    """
    Mengecek apakah teks hanya berisi angka, mention, URL, atau simbol lain yang tidak bermakna.
    """
    if not text.strip():  # Jika input kosong setelah dihapus spasi
        return True
    if re.fullmatch(r'[\d\s\W]+', text):  # Jika hanya mengandung angka atau simbol
        return True
    return False

# Inisialisasi Streamlit untuk tampilan web aplikasi
st.title("Analisis Sentimen Kampus Merdeka")  # Judul aplikasi
st.write("Masukkan teks untuk dianalisis:")  # Petunjuk input teks

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks di sini", "")

# Tombol untuk melakukan analisis sentimen
if st.button("Analisis Sentimen"):
    if user_input:  # Pastikan input tidak kosong
        if is_invalid_input(user_input):  # Cek apakah input tidak valid
            st.write("**Maaf, tidak bisa memprediksi angka, simbol, atau mention.**")
        else:
            # Langkah 1: Preprocessing teks
            cleaned_text = preprocess_text(user_input)
            cleaned_text = handle_negation(cleaned_text)  # Tangani negasi agar model lebih akurat
            
            # Langkah 2: Transformasi teks ke bentuk vektor TF-IDF
            text_tfidf = tfidf_vectorizer.transform([cleaned_text])
            
            # Langkah 3: Prediksi sentimen menggunakan model SVM
            prediction = svm_model.predict(text_tfidf)
            
            # Langkah 4: Mapping hasil prediksi (0 = Negatif, 1 = Positif)
            sentiment_map = {0: "Negatif", 1: "Positif"}  
            sentiment = sentiment_map.get(prediction[0], "Tidak diketahui")  # Jika tidak dikenali, tampilkan "Tidak diketahui"

            # Langkah 5: Menampilkan hasil prediksi ke pengguna
            st.write(f"Hasil analisis sentimen: **{sentiment}**")
    else:
        st.write("Silakan masukkan teks untuk dianalisis.")  # Pesan jika input kosong

# Tambahkan nama di bagian bawah
st.write("\n\n---\n**Hanun Salsabila 21.11.4144**")
