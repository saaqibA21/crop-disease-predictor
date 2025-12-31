# src/infer_manual.py

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from cure_guide import CURE_GUIDE, DEFAULT_CURE

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hybrid_manual_efficientnet.h5"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map_manual.npy"
CSV_PATH = BASE_DIR / "data" / "dataset.csv"

IMG_SIZE = (256, 256)
CONTEXT_COLS = ["crop", "region", "season"]

_model = None
_idx_to_class = None
_ctx_encoder = None


def _load_label_map():
    global _idx_to_class
    if _idx_to_class is None:
        _idx_to_class = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    return _idx_to_class


def _load_context_encoder():
    global _ctx_encoder
    if _ctx_encoder is None:
        df = pd.read_csv(CSV_PATH)
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit(df[CONTEXT_COLS])
        _ctx_encoder = enc
    return _ctx_encoder


def _load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def infer(image: Image.Image, crop: str, region: str, season: str) -> Dict:
    model = _load_model()
    idx_to_class = _load_label_map()
    ctx_encoder = _load_context_encoder()

    img_batch = _preprocess_image(image)

    ctx_df = pd.DataFrame([{
        "crop": crop,
        "region": region,
        "season": season,
    }])
    ctx_vec = ctx_encoder.transform(ctx_df[CONTEXT_COLS])

    preds = model.predict([img_batch, ctx_vec], verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = idx_to_class[idx]

    cure_info = CURE_GUIDE.get(label, DEFAULT_CURE)
    issue = cure_info["issue"]
    cure = cure_info["cure"]

    return {
        "label": label,
        "issue": issue,
        "confidence": confidence,
        "cure": cure,
    }
