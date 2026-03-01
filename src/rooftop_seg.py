import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
DATA_LIMIT = 15000
IMAGE_DIR = "./data/outputs/roofs/images"
MASK_DIR = "./data/outputs/roofs/masks"
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# DATASET
# -------------------------------
class RoofDataset(Dataset):
    def __init__(self, image_dir, mask_dir, data_limit):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        image_names = {
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
        }
        mask_names = {
            f for f in os.listdir(mask_dir)
            if os.path.isfile(os.path.join(mask_dir, f))
        }
        common = sorted(image_names.intersection(mask_names))
        self.images = common[:min(data_limit, len(common))]

        if not self.images:
            raise ValueError(
                f"No matching image/mask files found in '{image_dir}' and '{mask_dir}'."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.images):
            raise IndexError(
                f"Index {idx} out of range for dataset size {len(self.images)}"
            )

        name = self.images[idx]
        image_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0  # 0 or 1

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# -------------------------------
# MODEL (CUSTOM, NOT U-NET)
# -------------------------------
class SimpleRoofSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x1 = self.enc1(x)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        x = self.pool3(x3)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return x

# -------------------------------
# TRAINING
# -------------------------------
dataset = RoofDataset(IMAGE_DIR, MASK_DIR, DATA_LIMIT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleRoofSegNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Training on {DEVICE}")

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "./roof_seg_model.pth")

# -------------------------------
# VISUALIZATION
# -------------------------------
model.eval()

image, mask = dataset[0]
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(DEVICE))
    pred = torch.sigmoid(pred)          # Tensor
    pred = pred.cpu().squeeze().numpy() # NumPy

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image.permute(1, 2, 0))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(mask.squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred > 0.4, cmap="gray") 
plt.axis("off")

import random
for i in range(3):
    image, mask = dataset[random.randint(0, len(dataset) - 1)]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(DEVICE))
        pred = torch.sigmoid(pred)          # Tensor
        pred = pred.cpu().squeeze().numpy() # NumPy

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred > 0.4, cmap="gray")   # ✅ FIXED
    plt.axis("off")


plt.tight_layout()
plt.show()
