import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------
# Config
# -------------------------
MODEL_PATH = "model/best_model_stage2.h5"
IMG_SIZE = 224

# Your confirmed mapping:
# {'covid': 0, 'normal': 1}
# Model output is sigmoid -> P(class=1) = P(NORMAL)
CLASS_INDEX_TO_NAME = {0: "COVID", 1: "NORMAL"}

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(pil_img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x

st.set_page_config(page_title="COVID-19 X-ray Classifier", layout="centered")
st.title("COVID-19 X-ray Classifier (ResNet50)")
st.write("Upload a chest X-ray image and get a prediction.")

uploaded = st.file_uploader("Upload an X-ray image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    model = get_model()
    x = preprocess_image(img)

    prob_normal = float(model.predict(x, verbose=0)[0][0])  # P(class=1) = P(NORMAL)
    prob_covid = 1.0 - prob_normal

    pred_idx = 1 if prob_normal >= 0.5 else 0
    pred_label = CLASS_INDEX_TO_NAME[pred_idx]

    st.subheader("Prediction")
    st.write(f"**{pred_label}**")

    st.subheader("Probabilities")
    st.write(f"COVID probability: **{prob_covid:.4f}**")
    st.write(f"Normal probability: **{prob_normal:.4f}**")

    st.caption("Note: This is a project demo, not a medical diagnostic tool.")
