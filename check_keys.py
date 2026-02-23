
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append('src')
from cure_guide import CURE_GUIDE

def check_keys():
    label_map_path = Path('models/label_map_manual.npy')
    if not label_map_path.exists():
        print("Model map not found")
        return
        
    idx_to_class = np.load(label_map_path, allow_pickle=True).item()
    labels = set(idx_to_class.values())
    guide_keys = set(CURE_GUIDE.keys())
    
    missing = labels - guide_keys
    if missing:
        print(f"MISSING KEYS in CURE_GUIDE: {missing}")
    else:
        print("All labels are present in CURE_GUIDE")
        
    # Check for extra keys in guide (typos)
    extra = guide_keys - labels
    if extra:
        print(f"EXTRA KEYS in CURE_GUIDE (possible typos): {extra}")

if __name__ == "__main__":
    check_keys()
