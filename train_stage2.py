import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------
# Configuration
# -------------------------

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10   # fine-tuning usually needs fewer epochs

MODEL_DIR = "model"
STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
STAGE2_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_stage2.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Data Generators
# -------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -------------------------
# Load Stage-1 Best Model
# -------------------------

print("📦 Loading Stage-1 best model...")
model = load_model(STAGE1_MODEL_PATH)

# -------------------------
# Fine-tuning: Unfreeze last layers
# -------------------------

# Freeze everything first
for layer in model.layers:
    layer.trainable = False

# Unfreeze last 30 layers (skip BatchNorm for stability)
for layer in model.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# -------------------------
# Recompile with lower LR
# -------------------------

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Callbacks
# -------------------------

checkpoint = ModelCheckpoint(
    filepath=STAGE2_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# -------------------------
# Train (Fine-tuning)
# -------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("✅ Fine-tuning complete")
print(f"✅ Best fine-tuned model saved at: {STAGE2_MODEL_PATH}")
