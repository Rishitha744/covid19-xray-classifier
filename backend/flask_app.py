import io
import os
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_model_stage2.h5")

IMG_SIZE = 224
THRESHOLD = 0.5

# Your confirmed mapping: {'covid': 0, 'normal': 1}
# Sigmoid output = P(class=1) = P(NORMAL)

app = Flask(__name__)
CORS(app)

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data key 'file'."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_bytes = f.read()
    x = preprocess_image(file_bytes)

    prob_normal = float(model.predict(x, verbose=0)[0][0])  # P(NORMAL)
    prob_covid = 1.0 - prob_normal

    label = "NORMAL" if prob_normal >= THRESHOLD else "COVID"

    return jsonify({
        "label": label,
        "prob_covid": round(prob_covid, 4),
        "prob_normal": round(prob_normal, 4),
        "threshold": THRESHOLD
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
