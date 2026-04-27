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
# Point to the new high-performance model
MODEL_PATH = BASE_DIR / "models" / "hybrid_v3_fast.keras"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map_v3.npy"
CSV_PATH = BASE_DIR / "data" / "dataset.csv"

IMG_SIZE = (224, 224)
CONTEXT_COLS = ["crop", "region", "season"]

_model = None
_idx_to_class = None
_ctx_encoder = None


def _load_label_map():
    global _idx_to_class
    if _idx_to_class is None:
        if not LABEL_MAP_PATH.exists():
            # Fallback to old label map if new one doesn't exist yet
            old_path = BASE_DIR / "models" / "label_map_manual.npy"
            if old_path.exists():
                _idx_to_class = np.load(old_path, allow_pickle=True).item()
            else:
                raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}")
        else:
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
        if not MODEL_PATH.exists():
            # Fallback to old model if new one doesn't exist yet
            old_model_path = BASE_DIR / "models" / "hybrid_manual_efficientnet.h5"
            if old_model_path.exists():
                print(f"⚠️ New model not found, falling back to {old_model_path.name}")
                _model = tf.keras.models.load_model(old_model_path)
            else:
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        else:
            _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    # MobileNetV3 handles its own scaling/preprocessing internally if built with Keras applications
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
    ctx_vec = ctx_encoder.transform(ctx_df[CONTEXT_COLS]).astype("float32")

    # Use a dictionary for inputs to be compatible with Keras 3
    # and ensure inputs are tensors
    inputs = {
        "image": tf.convert_to_tensor(img_batch),
        "context": tf.convert_to_tensor(ctx_vec)
    }
    preds = model.predict(inputs, verbose=0)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(preds)[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        p_label = idx_to_class[idx]
        p_conf = float(preds[idx])
        
        display_name = p_label.replace("___", " - ").replace("_", " ").title()
        
        if p_label in CURE_GUIDE:
            p_cure_info = CURE_GUIDE[p_label]
            issue_name = p_cure_info["issue"]
            cure_text = p_cure_info["cure"]
        else:
            issue_name = display_name
            cure_text = DEFAULT_CURE["cure"]
            
        top_predictions.append({
            "label": p_label,
            "confidence": p_conf,
            "issue": issue_name,
            "cure": cure_text
        })

    best = top_predictions[0]

    return {
        "label": best["label"],
        "issue": best["issue"],
        "confidence": best["confidence"],
        "cure": best["cure"],
        "top_predictions": top_predictions
    }
