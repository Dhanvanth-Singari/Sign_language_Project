from PIL import Image
from pathlib import Path
import numpy as np

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def sorted_image_paths(dir_path):
    p = Path(dir_path)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    return files
