
import cv2
import numpy as np

def extract_valid_components(binary_mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8 
    )

    components = []

    for label in range(1, num_labels):  
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component = (labels == label).astype(np.uint8) * 255
        components.append(component)

    return components

def crop_with_context(image, full_mask, component_mask, context_ratio=0.2):
    x, y, w, h = cv2.boundingRect(component_mask)

    pad_w = int(w * context_ratio)
    pad_h = int(h * context_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)

    crop_img = image[y1:y2, x1:x2]
    crop_mask = full_mask[y1:y2, x1:x2]

    return crop_img, crop_mask

def resize_with_padding(image, mask, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    image_resized = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image_padded = cv2.copyMakeBorder(
        image_resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    mask_padded = cv2.copyMakeBorder(
        mask_resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    return image_padded, mask_padded