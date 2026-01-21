import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------
# Configuration
# -------------------------
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20   # early stopping will stop earlier if needed

MODEL_DIR = "model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Data Generators
# -------------------------

# Training data (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

# Validation data (NO augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

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
# Model: ResNet50 (Transfer Learning)
# -------------------------

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Callbacks
# -------------------------

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
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
# Train the Model
# -------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# -------------------------
# Save Final Model
# -------------------------

model.save(LAST_MODEL_PATH)

print("✅ Training complete")
print(f"✅ Best model saved at: {BEST_MODEL_PATH}")
print(f"✅ Last model saved at: {LAST_MODEL_PATH}")

# -------------------------
# Print Best Epoch Metrics
# -------------------------

best_epoch = checkpoint.best_epoch if hasattr(checkpoint, "best_epoch") else None

if best_epoch is not None:
    best_val_acc = max(history.history["val_accuracy"])
    best_train_acc = history.history["accuracy"][history.history["val_accuracy"].index(best_val_acc)]

    print("\n📌 Best Epoch Summary")
    print(f"Best Epoch: {history.history['val_accuracy'].index(best_val_acc) + 1}")
    print(f"Training Accuracy (Best Epoch): {best_train_acc:.4f}")
    print(f"Validation Accuracy (Best Epoch): {best_val_acc:.4f}")
