
import os
import cv2
import numpy as np
from tqdm import tqdm
from paths import TRAIN_IMG_DIR, TRAIN_MASK_DIR
from utils import (
    extract_valid_components,
    crop_with_context,
    resize_with_padding
)

IMG_DIR = TRAIN_IMG_DIR
MASK_DIR = TRAIN_MASK_DIR

OUT_IMG_DIR = "./data/outputs/roofs/images"
OUT_MASK_DIR = "./data/outputs/roofs/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

TARGET_SIZE = 256
MIN_AREA_PIXELS = 500  
CONTEXT_RATIO = 0.2    

roof_counter = 0

image_files = sorted(os.listdir(IMG_DIR))

for img_name in tqdm(image_files, desc="Extracting rooftops (clean)"):
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Shape mismatch: {img_name}")

    mask = (mask > 127).astype(np.uint8) * 255

    components = extract_valid_components(
        mask,
        min_area=MIN_AREA_PIXELS
    )

    for component in components:
        crop_img, crop_mask = crop_with_context(
            image,
            mask,
            component,
            context_ratio=CONTEXT_RATIO
        )

        crop_img, crop_mask = resize_with_padding(
            crop_img,
            crop_mask,
            TARGET_SIZE
        )

        cv2.imwrite(
            f"{OUT_IMG_DIR}/roof_{roof_counter:06d}.png",
            crop_img
        )
        cv2.imwrite(
            f"{OUT_MASK_DIR}/roof_{roof_counter:06d}.png",
            crop_mask
        )

        roof_counter += 1

print(f"✅ Clean rooftops extracted: {roof_counter}")