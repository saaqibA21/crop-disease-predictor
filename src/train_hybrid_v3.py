# src/train_hybrid_v3.py
"""
High-Performance Hybrid Model Training Script.
Target: 95%+ Accuracy, Fast Inference.
Architecture: MobileNetV3Large + Context Fusion.
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_v3_fast.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map_v3.npy"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FREEZED = 5
EPOCHS_FINE_TUNE = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4

CONTEXT_COLS = ["crop", "region", "season"]

# -------------------------------------------------------------------
# DATA LOADING & PIPELINE
# -------------------------------------------------------------------

def load_and_preprocess_image(path, label, context):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # MobileNetV3 expects pixels in [0, 255] for its internal preprocessing 
    # but we will use the rescale layer in the model for consistency.
    return (img, context), label

def augment(img, context, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return (img, context), label

def build_dataset(df, encoder, class_to_idx, is_training=True):
    paths = df["image_path"].values
    labels = tf.keras.utils.to_categorical(df["label"].map(class_to_idx).values, num_classes=len(class_to_idx))
    contexts = encoder.transform(df[CONTEXT_COLS]).astype("float32")
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, contexts))
    
    # Map the loading function
    ds = ds.map(lambda p, l, c: load_and_preprocess_image(p, l, c), num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(buffer_size=2000)
        ds = ds.map(lambda x_y, l: augment(x_y[0], x_y[1], l), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------------------------------------------------
# MODEL ARCHITECTURE
# -------------------------------------------------------------------

def build_hybrid_v3(num_classes, num_ctx_features):
    # Image Input
    img_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    
    # Pre-trained base
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base_model.trainable = False
    
    x_img = base_model(img_input)
    x_img = layers.Dense(256, activation="relu")(x_img)
    x_img = layers.Dropout(0.3)(x_img)
    
    # Context Input
    ctx_input = layers.Input(shape=(num_ctx_features,), name="context")
    x_ctx = layers.Dense(64, activation="relu")(ctx_input)
    x_ctx = layers.BatchNormalization()(x_ctx)
    
    # Fusion
    combined = layers.Concatenate()([x_img, x_ctx])
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)
    combined = layers.Dense(128, activation="relu")(combined)
    
    output = layers.Dense(num_classes, activation="softmax")(combined)
    
    model = models.Model(inputs=[img_input, ctx_input], outputs=output)
    return model, base_model

# -------------------------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------------------------

def main():
    print("Starting High-Performance Training...")
    
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    np.save(LABEL_MAP_PATH, idx_to_class)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[CONTEXT_COLS])
    num_ctx_features = encoder.transform(df.iloc[:1][CONTEXT_COLS]).shape[1]
    
    # Split
    df_train, df_val = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    
    train_ds = build_dataset(df_train, encoder, class_to_idx, is_training=True)
    val_ds = build_dataset(df_val, encoder, class_to_idx, is_training=False)
    
    # 2. Build Model
    model, base_model = build_hybrid_v3(len(classes), num_ctx_features)
    
    # 3. Phase 1: Warm up (Frozen Base)
    print("\nPhase 1: Warming up top layers...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FREEZED,
        callbacks=[
            callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            checkpoint,
            lr_reducer
        ]
    )
    
    # 4. Phase 2: Fine-tuning (Unfrozen Base)
    print("\nPhase 2: Fine-tuning entire network...")
    base_model.trainable = True
    # Optional: freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks are already defined above
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=[checkpoint, lr_reducer]
    )
    
    print(f"\nTraining Complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
