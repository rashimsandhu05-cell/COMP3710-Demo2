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

DATA_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_train"
OUTPUT_DIR = "vae_results"

IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_DIM = 2
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

class OASISDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))

        if len(self.files) == 0:
            raise RuntimeError(f"No PNG files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        return image


# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ============================================================
# Loss
# ============================================================

def vae_loss(reconstruction, target, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        target,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss


# ============================================================
# Training
# ============================================================

dataset = OASISDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Training images:", len(dataset))
print("Batch size:", BATCH_SIZE)
print("Image size:", IMAGE_SIZE)
print("Latent dimensions:", LATENT_DIM)
print("Epochs:", EPOCHS)


model = VAE(LATENT_DIM).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


loss_history = []

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for images in loader:

        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(images)

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataset)
    loss_history.append(average_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# ============================================================
# Save model
# ============================================================

model_path = os.path.join(OUTPUT_DIR, "oasis_vae.pth")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "latent_dim": LATENT_DIM,
        "image_size": IMAGE_SIZE,
        "loss_history": loss_history
    },
    model_path
)

print("Model saved to:", model_path)
