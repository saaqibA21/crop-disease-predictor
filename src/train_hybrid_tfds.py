# src/train_hybrid_tfds.py
# src/train_hybrid_tfds.py

import numpy as np
from pathlib import Path
import os  # ← add this line

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense,
    Concatenate,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

# Force TFDS to use a short data directory to avoid long Windows paths
os.environ["TFDS_DATA_DIR"] = r"C:\tfds"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_plantvillage_efficientnet.h5"
LABEL_NAMES_PATH = MODEL_DIR / "label_names.npy"
CROP_NAMES_PATH = MODEL_DIR / "crop_names.npy"

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15

AUTOTUNE = tf.data.AUTOTUNE


def build_label_and_crop_mappings(label_names):
    """
    From PlantVillage label names like 'Tomato___Early_blight', build:
    - label_names: list of all class names
    - crop_names: list of unique crop names (tomato, potato, etc.)
    - label_to_crop_index: np.array mapping label_id -> crop_id
    """
    # Extract crop names by splitting at '___'
    crops = []
    for name in label_names:
        crop = name.split("___")[0].strip()
        crops.append(crop)

    crop_names = sorted(set(crops))
    crop_to_idx = {c: i for i, c in enumerate(crop_names)}

    label_to_crop_index = np.array(
        [crop_to_idx[name.split("___")[0].strip()] for name in label_names],
        dtype=np.int32,
    )

    return crop_names, label_to_crop_index


def prepare_datasets():
    """
    Loads PlantVillage using TFDS, builds mapping, and returns:
    - train_ds, val_ds
    - num_classes, num_crops
    - label_names, crop_names
    """
    (ds_train_raw, ds_val_raw), ds_info = tfds.load(
    "plant_village",
    split=["train[:85%]", "train[85%:]"],
    as_supervised=True,
    with_info=True,
    data_dir=r"C:\tfds",
)


    num_classes = ds_info.features["label"].num_classes
    label_names = ds_info.features["label"].names

    crop_names, label_to_crop_index = build_label_and_crop_mappings(label_names)
    num_crops = len(crop_names)

    # Make a TF constant so we can use it in map()
    label_to_crop_index_tf = tf.constant(label_to_crop_index, dtype=tf.int32)

    def preprocess(image, label):
        # Resize & scale image
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0

        # Label -> one-hot
        label_one_hot = tf.one_hot(label, depth=num_classes)

        # Build simple context: crop one-hot based on label
        crop_idx = tf.gather(label_to_crop_index_tf, label)
        crop_one_hot = tf.one_hot(crop_idx, depth=num_crops)

        features = {
            "image": image,
            "context": crop_one_hot,
        }
        return features, label_one_hot

    def augment(features, label):
        img = features["image"]
        # Simple augmentations for robustness
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        features["image"] = img
        return features, label

    train_ds = ds_train_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = ds_val_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return (
        train_ds,
        val_ds,
        num_classes,
        num_crops,
        label_names,
        crop_names,
    )


def build_hybrid_model(num_classes, num_crops):
    """
    Image branch: EfficientNetB3
    Context branch: crop one-hot
    Fusion → Dense → softmax
    """
    # Image input
    img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    base_cnn = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input,
    )
    x_img = GlobalAveragePooling2D()(base_cnn.output)
    x_img = Dense(256, activation="relu")(x_img)

    # Context input (crop one-hot)
    ctx_input = Input(shape=(num_crops,), name="context")
    x_ctx = Dense(64, activation="relu")(ctx_input)

    # Fusion
    x = Concatenate()([x_img, x_ctx])
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[img_input, ctx_input], outputs=output)

    # Freeze most EfficientNet layers first, fine-tune later if needed
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False

    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading PlantVillage with TFDS…")
    (
        train_ds,
        val_ds,
        num_classes,
        num_crops,
        label_names,
        crop_names,
    ) = prepare_datasets()

    print(f"Number of disease classes: {num_classes}")
    print(f"Crop types (for context branch): {crop_names}")

    print("Building hybrid EfficientNet model…")
    model = build_hybrid_model(num_classes, num_crops)
    model.summary()

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

    print("Starting training…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("Training complete.")
    print(f"Best model saved to: {MODEL_PATH}")

    # Save label & crop names for inference
    np.save(LABEL_NAMES_PATH, np.array(label_names))
    np.save(CROP_NAMES_PATH, np.array(crop_names))
    print(f"Saved label names to: {LABEL_NAMES_PATH}")
    print(f"Saved crop names to: {CROP_NAMES_PATH}")


if __name__ == "__main__":
    main()
