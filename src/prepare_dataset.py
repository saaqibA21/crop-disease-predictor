# src/prepare_dataset.py

from pathlib import Path
import pandas as pd
import random

BASE_DIR = Path(__file__).resolve().parent.parent
# We use only the color images
IMAGE_DIR = BASE_DIR / "data" / "images" / "plantvillage_dataset" / "color"
CSV_PATH = BASE_DIR / "data" / "dataset.csv"

REGIONS = ["Tamil Nadu", "Karnataka", "Maharashtra", "Uttar Pradesh", "Punjab"]
SEASONS = ["Kharif", "Rabi", "Summer"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def infer_crop_from_class(class_name: str) -> str:
    # PlantVillage class names like "Tomato___Early_blight"
    return class_name.split("___")[0]


def main():
    if not IMAGE_DIR.exists():
        print(f"❌ Image directory does not exist: {IMAGE_DIR}")
        return

    rows = []

    # Recursively go through all class folders inside color/
    for img_file in IMAGE_DIR.rglob("*"):
        if img_file.suffix not in IMG_EXTS:
            continue

        # Parent folder is the class name, e.g. "Tomato___Early_blight"
        class_name = img_file.parent.name

        crop = infer_crop_from_class(class_name)
        region = random.choice(REGIONS)
        season = random.choice(SEASONS)

        rows.append({
            "image_path": str(img_file),
            "label": class_name,
            "crop": crop,
            "region": region,
            "season": season,
        })

    print(f"Found {len(rows)} image files under {IMAGE_DIR}")

    if not rows:
        print("⚠️ No images found. Check that your images are inside data/images/color/CLASS_NAME/*.jpg")
        return

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Saved dataset CSV with {len(df)} rows at {CSV_PATH}")


if __name__ == "__main__":
    main()
