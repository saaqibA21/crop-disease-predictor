
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from infer_manual import infer

def test_single_image():
    # Path to a known image (Tomato Early Blight)
    test_img_path = BASE_DIR / "data" / "images" / "plantvillage_dataset" / "color" / "Tomato___Early_blight" / "0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG"
    
    if not test_img_path.exists():
        print(f"ERROR: Test image not found at {test_img_path}")
        return

    print(f"Testing with image: {test_img_path.name}")
    image = Image.open(test_img_path)
    
    # Run inference
    try:
        result = infer(
            image,
            crop="Tomato",
            region="Tamil Nadu",
            season="Kharif"
        )
        
        print("\n--- INFERENCE RESULTS ---")
        print(f"Detected Label: {result['label']}")
        print(f"Issue Name:     {result['issue']}")
        print(f"Confidence:     {result['confidence']:.4f}")
        print(f"Cure Provided:  {result['cure'][:100]}...")
        
        print("\nTop 3 Predictions:")
        for i, p in enumerate(result['top_predictions']):
            print(f"{i+1}. {p['issue']} ({p['confidence']:.4f})")
            
    except Exception as e:
        print(f"CRITICAL ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_image()
