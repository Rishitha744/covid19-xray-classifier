import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 16

MODEL_PATH = "model/best_model_stage2.h5"
TEST_DIR = "data/test"

datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = load_model(MODEL_PATH)

# Predict
probs = model.predict(test_generator)
y_pred = (probs >= 0.5).astype(int).ravel()
y_true = test_generator.classes

# Metrics
acc = accuracy_score(y_true, y_pred)
print("\n Test Accuracy:", round(acc * 100, 2), "%")

print("\n Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\n Classification Report")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["covid", "normal"]
    )
)
print("Class indices:", test_generator.class_indices)
