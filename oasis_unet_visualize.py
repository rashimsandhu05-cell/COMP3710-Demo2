import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

IMAGE_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_test"
MASK_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_seg_test"

MODEL_PATH = "unet_results/oasis_unet_best.pth"
OUTPUT_DIR = "unet_results"
IMAGE_SIZE = 128
NUM_CLASSES = 4


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


class OASISTestDataset(Dataset):
    def __init__(self):
        self.images = sorted([
            f for f in os.listdir(IMAGE_DIR)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]

        image_path = os.path.join(IMAGE_DIR, filename)

        image = Image.open(image_path).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        return image, filename


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = UNet(NUM_CLASSES).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

dataset = OASISTestDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)

images, filenames = next(iter(loader))

images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)

images = images.cpu().numpy()
predictions = predictions.cpu().numpy()

fig, axes = plt.subplots(4, 3, figsize=(10, 12))

for i in range(4):
    axes[i, 0].imshow(images[i, 0], cmap="gray")
    axes[i, 0].set_title("Original MRI")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(predictions[i], cmap="viridis", vmin=0, vmax=3)
    axes[i, 1].set_title("Predicted Segmentation")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(images[i, 0], cmap="gray")
    axes[i, 2].imshow(
        predictions[i],
        cmap="viridis",
        alpha=0.45,
        vmin=0,
        vmax=3
    )
    axes[i, 2].set_title("Overlay")
    axes[i, 2].axis("off")

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "unet_test_predictions.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print("Saved:", output_path)
