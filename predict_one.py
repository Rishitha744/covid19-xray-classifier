import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model/best_model_stage2.h5"
IMG_SIZE = 224

def predict_image(img_path):
    model = load_model(MODEL_PATH)

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    prob = float(model.predict(x, verbose=0)[0][0])  # probability of class=1
    label = "NORMAL" if prob >= 0.5 else "COVID"

    print("Image:", img_path)
    print("Normal probability:", round(prob, 4))
    print("Prediction:", label)

if __name__ == "__main__":
    # ✅ Put an actual image path from your folder here
    predict_image("data/test/covid/COVID-100.png")
