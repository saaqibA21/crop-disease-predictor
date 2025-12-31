# src/train_hybrid_manual.py
"""
Hybrid EfficientNetB3 + context model trained on manual PlantVillage folders.

Uses:
- data/dataset.csv (built by src/prepare_dataset.py)
- images from data/images/color/CLASS_NAME/*.jpg

Saves:
- models/hybrid_manual_efficientnet.h5
- models/label_map_manual.npy
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_manual_efficientnet.h5"
LABEL_MAP_PATH = MODEL_DIR / "label_map_manual.npy"

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 15

CONTEXT_COLS = ["crop", "region", "season"]


# -------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------
def load_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run src/prepare_dataset.py after placing images "
            "inside data/images/color/CLASS_NAME/*.jpg"
        )
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError(
            f"{CSV_PATH} is empty. Ensure prepare_dataset.py found images "
            "and re-run it."
        )
    return df


def build_label_encoder(df: pd.DataFrame):
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    df = df.copy()
    df["label_idx"] = df["label"].map(class_to_idx)
    return df, class_to_idx, idx_to_class


def build_context_encoder(df: pd.DataFrame):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(df[CONTEXT_COLS])
    return enc


def split_dataset(df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=42):
    df_train, df_temp = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df["label"],
        random_state=random_state,
    )
    rel_val = val_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - rel_val),
        stratify=df_temp["label"],
        random_state=random_state,
    )
    return df_train, df_val, df_test


def data_generator(df, enc, class_to_idx, batch_size, img_size, augment=True):
    """
    Python generator that yields:
        ((images_batch, context_batch), labels_batch)
    where:
        images_batch: (B, H, W, 3) float32 in [0, 1]
        context_batch: (B, num_ctx_features) float32
        labels_batch: (B, num_classes) one-hot float32
    """
    df = df.reset_index(drop=True)
    n = len(df)

    while True:
        df = shuffle(df)
        ctx_encoded = enc.transform(df[CONTEXT_COLS]).astype("float32")
        num_classes = len(class_to_idx)

        for i in range(0, n, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_ctx = ctx_encoded[i:i + batch_size]

            imgs = []
            labels = []
            for _, row in batch_df.iterrows():
                img = tf.keras.preprocessing.image.load_img(
                    row["image_path"], target_size=img_size
                )
                img = tf.keras.preprocessing.image.img_to_array(img)
                imgs.append(img)
                labels.append(row["label_idx"])

            imgs = np.array(imgs, dtype="float32") / 255.0
            labels = tf.keras.utils.to_categorical(
                labels, num_classes=num_classes
            ).astype("float32")

            if augment:
                # Do simple tf.image augmentations
                imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
                imgs_tf = tf.image.random_flip_left_right(imgs_tf)
                imgs_tf = tf.image.random_brightness(imgs_tf, max_delta=0.08)
                imgs_tf = tf.image.random_contrast(imgs_tf, 0.9, 1.1)
                imgs = imgs_tf

            # IMPORTANT: return ((imgs, ctx), labels) – tuple, not list
            yield ( (imgs, batch_ctx), labels )


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
def build_hybrid_model(num_classes: int, num_ctx_features: int) -> tf.keras.Model:
    img_input = Input(
        shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image"
    )
    base_cnn = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input,
    )
    x_img = GlobalAveragePooling2D()(base_cnn.output)
    x_img = Dense(256, activation="relu")(x_img)

    ctx_input = Input(shape=(num_ctx_features,), name="context")
    x_ctx = Dense(128, activation="relu")(ctx_input)
    x_ctx = Dense(64, activation="relu")(x_ctx)

    x = Concatenate()([x_img, x_ctx])
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[img_input, ctx_input], outputs=output)

    # Freeze most of EfficientNet for speed & stability
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def main():
    df = load_dataset()
    df, class_to_idx, idx_to_class = build_label_encoder(df)

    # Save label map (index -> class name)
    np.save(LABEL_MAP_PATH, idx_to_class)
    print("Saved label map to", LABEL_MAP_PATH)

    enc = build_context_encoder(df)
    ctx_dummy = enc.transform(df[CONTEXT_COLS])
    num_ctx_features = ctx_dummy.shape[1]
    num_classes = len(class_to_idx)

    df_train, df_val, df_test = split_dataset(df)

    train_steps = max(1, math.ceil(len(df_train) / BATCH_SIZE))
    val_steps = max(1, math.ceil(len(df_val) / BATCH_SIZE))

    model = build_hybrid_model(num_classes, num_ctx_features)
    model.summary()

    # Explicitly wrap generators in tf.data.Dataset with a proper output_signature
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(
            df_train, enc, class_to_idx, BATCH_SIZE, IMG_SIZE, augment=True
        ),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(None, num_ctx_features),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(None, num_classes),
                dtype=tf.float32,
            ),
        ),
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(
            df_val, enc, class_to_idx, BATCH_SIZE, IMG_SIZE, augment=False
        ),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(None, num_ctx_features),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(None, num_classes),
                dtype=tf.float32,
            ),
        ),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    print("Training complete. Best model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
