# src/infer_tfds.py

from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image

from cure_guide import CURE_GUIDE, DEFAULT_CURE

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hybrid_plantvillage_efficientnet.h5"
LABEL_NAMES_PATH = BASE_DIR / "models" / "label_names.npy"
CROP_NAMES_PATH = BASE_DIR / "models" / "crop_names.npy"

IMG_SIZE = (256, 256)

_model = None
_label_names = None
_crop_names = None
_crop_to_idx = None


def _load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _load_label_and_crop_names():
    global _label_names, _crop_names, _crop_to_idx
    if _label_names is None:
        _label_names = np.load(LABEL_NAMES_PATH, allow_pickle=True)
    if _crop_names is None:
        _crop_names = np.load(CROP_NAMES_PATH, allow_pickle=True)
        _crop_to_idx = {c: i for i, c in enumerate(_crop_names)}
    return _label_names, _crop_names, _crop_to_idx


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def build_context_vector(crop_name: str) -> np.ndarray:
    _, crop_names, crop_to_idx = _load_label_and_crop_names()
    # Try to match case-insensitively
    norm = crop_name.strip().lower()
    found_key = None
    for c in crop_names:
        if c.lower() == norm:
            found_key = c
            break
    if found_key is None:
        # if unknown, we just use a zero vector (no extra info)
        return np.zeros((1, len(crop_names)), dtype="float32")

    idx = crop_to_idx[found_key]
    vec = np.zeros((len(crop_names),), dtype="float32")
    vec[idx] = 1.0
    return np.expand_dims(vec, axis=0)


def infer(image: Image.Image, crop: str) -> Dict:
    """
    image: PIL Image
    crop: crop type string (e.g. 'Tomato', 'Potato')
    Returns: dict with label, issue, confidence, cure
    """
    model = _load_model()
    label_names, _, _ = _load_label_and_crop_names()

    img_batch = preprocess_image(image)
    ctx_batch = build_context_vector(crop)

    # Model expects inputs named "image" and "context"
    preds = model.predict({"image": img_batch, "context": ctx_batch}, verbose=0)[0]

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = str(label_names[idx])

    cure_info = CURE_GUIDE.get(label, DEFAULT_CURE)
    issue = cure_info["issue"]
    cure = cure_info["cure"]

    return {
        "label": label,
        "issue": issue,
        "confidence": confidence,
        "cure": cure,
    }
