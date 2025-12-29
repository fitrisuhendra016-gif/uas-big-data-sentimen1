import streamlit as st
import os
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path ke src/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # path ke root repo
APP_FOLDER = os.path.join(ROOT_DIR, "app_sentimen")

model_path = os.path.join(APP_FOLDER, "bilstm_model.keras")
tokenizer_path = os.path.join(APP_FOLDER, "tokenizer.pkl")


# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(model_path, compile=False)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ===============================
# PARAMETER
# ===============================
MAX_LEN = 100


# ===============================
# STREAMLIT UI
# ===============================
st.title("📊 Prediksi Sentimen WhatsApp")

text = st.text_area("Masukkan teks", height=150)

if st.button("Prediksi"):
    if text.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        pred = model.predict(padded, verbose=0)[0][0]
        label = "Positif 😊" if pred >= 0.5 else "Negatif 😞"

        st.success(f"Hasil Prediksi: **{label}**")
