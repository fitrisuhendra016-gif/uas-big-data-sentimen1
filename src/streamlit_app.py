from tensorflow.keras.models import load_model
import pickle
import os
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FOLDER = os.path.join(BASE_DIR, "..", "app_sentimen")

model_path = os.path.join(APP_FOLDER, "bilstm_model.keras")
tokenizer_path = os.path.join(APP_FOLDER, "tokenizer.pkl")

# ⬇️ INI YANG NAMANYA LOAD MODEL
model = load_model(model_path, compile=False)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
