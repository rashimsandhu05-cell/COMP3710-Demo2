import os
import glob
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_train"
MASK_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"

VAL_IMAGE_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_validate"
VAL_MASK_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate"

OUTPUT_DIR = "unet_results"

IMAGE_SIZE = 128
NUM_CLASSES = 4

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
NUM_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

class OASISSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_files = sorted(
            glob.glob(
                os.path.join(image_dir, "*.png")
            )
        )

        self.mask_files = sorted(
            glob.glob(
                os.path.join(mask_dir, "*.png")
            )
        )

        if len(self.image_files) != len(self.mask_files):
            raise RuntimeError(
                f"Image/mask mismatch: "
                f"{len(self.image_files)} images, "
                f"{len(self.mask_files)} masks"
            )

        if len(self.image_files) == 0:
            raise RuntimeError(
                "No PNG files found."
            )

        print(
            f"Loaded {len(self.image_files)} image/mask pairs"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image = Image.open(
            self.image_files[index]
        ).convert("L")

        mask = Image.open(
            self.mask_files[index]
        ).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        mask = np.asarray(
            mask,
            dtype=np.uint8
        )

        # Convert mask pixel values:
        # 0   -> class 0
        # 85  -> class 1
        # 170 -> class 2
        # 255 -> class 3

        mask = np.rint(
            mask.astype(np.float32) / 85.0
        ).astype(np.int64)

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(mask)

        return image, mask


# ============================================================
# U-Net building blocks
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.output = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# Dice score
# ============================================================

def dice_scores(prediction, target):

    scores = []

    for class_id in range(NUM_CLASSES):

        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        intersection = (
            pred_class & target_class
        ).sum().item()

        pred_count = pred_class.sum().item()
        target_count = target_class.sum().item()

        denominator = (
            pred_count + target_count
        )

        if denominator == 0:
            score = 1.0
        else:
            score = (
                2.0 * intersection
                / denominator
            )

        scores.append(score)

    return scores


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    class_scores = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            scores = dice_scores(
                predictions,
                masks
            )

            for class_id in range(NUM_CLASSES):
                class_scores[class_id].append(
                    scores[class_id]
                )

    return [
        float(np.mean(scores))
        for scores in class_scores
    ]


# ============================================================
# Create datasets
# ============================================================

train_dataset = OASISSegmentationDataset(
    IMAGE_DIR,
    MASK_DIR
)

val_dataset = OASISSegmentationDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)


# ============================================================
# Model
# ============================================================

model = UNet(
    num_classes=NUM_CLASSES
).to(device)

print(
    "Model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_mean_dice = 0.0

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    train_loss = (
        running_loss
        / len(train_dataset)
    )

    validation_scores = evaluate(
        model,
        val_loader
    )

    mean_dice = float(
        np.mean(validation_scores)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {train_loss:.4f} "
        f"- Background DSC: {validation_scores[0]:.4f} "
        f"- Label 1 DSC: {validation_scores[1]:.4f} "
        f"- Label 2 DSC: {validation_scores[2]:.4f} "
        f"- Label 3 DSC: {validation_scores[3]:.4f} "
        f"- Mean DSC: {mean_dice:.4f}"
    )

    if mean_dice > best_mean_dice:

        best_mean_dice = mean_dice

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "num_classes":
                    NUM_CLASSES,
                "image_size":
                    IMAGE_SIZE,
                "best_mean_dice":
                    best_mean_dice,
                "validation_dice":
                    validation_scores
            },
            os.path.join(
                OUTPUT_DIR,
                "oasis_unet_best.pth"
            )
        )

        print(
            "  Saved new best model."
        )


training_time = (
    time.time() - start_time
)

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds"
)

print(
    f"Best mean DSC: "
    f"{best_mean_dice:.4f}"
)
