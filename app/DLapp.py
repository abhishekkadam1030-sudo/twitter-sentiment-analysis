# app.py
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load tokenizer and model ---
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("lstm_baseline.h5")   # or "lstm_baseline" if you used SavedModel format

# --- Parameters ---
MAXLEN = 40

import re

def clean_text(text: str) -> str:
    """Basic text cleaning to match training preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    return text




def preprocess_and_predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    probs = model.predict(padded)
    label = int(np.argmax(probs, axis=1)[0])   # 0 or 1
    return label

# --- Streamlit UI ---
st.title("📊 Sentiment Classification Demo")
user_input = st.text_area("Enter text to analyze:", "")

import numpy as np

if st.button("Predict"):
    if user_input.strip():
        label = preprocess_and_predict(user_input)  # should be int 0 or 1

        st.write(f"**Predicted label:** {label}")

        if label == 1:
            st.success("Positive 😀")
        elif label == 0:
            st.error("Negative 😟")
        else:
            st.warning(f"Unexpected label: {label!r}")
    else:
        st.warning("Please enter some text first.")


        