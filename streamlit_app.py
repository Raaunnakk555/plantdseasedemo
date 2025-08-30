import io
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector (Demo)", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detection — Demo App")
st.write("Upload a leaf image to get a disease prediction (stub) and remedies.")

api_url = st.secrets.get("API_URL", "http://localhost:8000")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API URL", api_url)
    st.caption("Default: http://localhost:8000")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Predicting…"):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("leaf.png", buf, "image/png")}
        try:
            r = requests.post(f"{api_url}/predict", files=files, timeout=20)
            if r.status_code == 200:
                data = r.json()
                st.success(f"Predicted: **{data['disease']}** (confidence: {data['confidence']:.2f})")
                st.subheader("Suggested Remedies")
                for i, step in enumerate(data.get("remedies", []), 1):
                    st.write(f"{i}. {step}")
            else:
                st.error(f"API error: {r.status_code} — {r.text}")
        except Exception as e:
            st.error(f"Failed to reach API at {api_url}. Error: {e}")
