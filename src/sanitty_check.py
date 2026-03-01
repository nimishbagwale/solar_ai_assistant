import cv2
import matplotlib.pyplot as plt
import os
import random

img_dir = "./data/outputs/roofs/images"
mask_dir = "./data/outputs/roofs/masks"

samples = random.sample(os.listdir(img_dir), 10)

for s in samples:
    img = cv2.imread(os.path.join(img_dir, s))
    mask = cv2.imread(os.path.join(mask_dir, s), 0)

    plt.figure(figsize=(4,4))
    plt.imshow(img[..., ::-1])
    plt.imshow(mask, alpha=0.4, cmap="Reds")
    plt.title(s)
    plt.axis("off")
    plt.show()