import streamlit as st
from PIL import Image
import numpy as np
import os, sqlite3, time
import pandas as pd

# -------------------------
# Load ML model directly (Keras/TensorFlow)
# -------------------------
try:
    import tensorflow as tf
except Exception as e:
    st.error("⚠ TensorFlow not found. Add tensorflow to requirements.txt.")
    st.stop()

MODEL_PATH = "app/backend/model/plant_disease_model.h5"  # adjust if needed
INPUT_SIZE = (224, 224)  # change if your model expects a different size

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Example labels (replace with your own)
CLASS_NAMES = ["Healthy", "Leaf Blight", "Rust", "Mildew"]

# -------------------------
# Preprocess & predict
# -------------------------
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img: Image.Image):
    arr = preprocess(img)
    preds = model.predict(arr)
    if preds.shape[-1] == 1:  # binary case
        prob = float(preds[0,0])
        label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
        confidence = prob if prob > 0.5 else 1 - prob
    else:  # multi-class case
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        confidence = float(np.max(preds))
    return label, confidence

# -------------------------
# History (SQLite, optional)
# -------------------------
@st.cache_resource
def get_db():
    conn = sqlite3.connect("predictions.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, filename TEXT, label TEXT, confidence REAL
        )
    """)
    conn.commit()
    return conn

def save_history(fname, label, conf):
    conn = get_db()
    conn.execute("INSERT INTO predictions (ts, filename, label, confidence) VALUES (?, ?, ?, ?)",
                 (time.time(), fname, label, conf))
    conn.commit()

def load_history(limit=50):
    conn = get_db()
    return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                             conn, params=(limit,))

# -------------------------
# Streamlit UI
# -------------------------
st.title("🌿 Plant Disease Detector (Standalone)")

file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(img)
        st.success(f"Prediction: *{label}* ({conf*100:.2f}% confidence)")
        save_history(file.name, label, conf)

if st.checkbox("Show Prediction History"):
    df = load_history()
    if df.empty:
        st.info("No history yet.")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")
