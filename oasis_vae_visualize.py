import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_train"
MODEL_PATH = "vae_results/oasis_vae.pth"
OUTPUT_DIR = "vae_results"

IMAGE_SIZE = 64
LATENT_DIM = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

class OASISDataset(Dataset):

    def __init__(self, data_dir):
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.png"))
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No PNG files found in {data_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = Image.open(
            self.files[index]
        ).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        return image


# ============================================================
# VAE MODEL
# ============================================================

class VAE(nn.Module):

    def __init__(self, latent_dim=2):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            256 * 4 * 4,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            256 * 4 * 4,
            latent_dim
        )

        self.decoder_input = nn.Linear(
            latent_dim,
            256 * 4 * 4
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x):

        x = self.encoder(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        return mu + epsilon * std

    def decode(self, z):

        x = self.decoder_input(z)

        x = x.view(
            -1,
            256,
            4,
            4
        )

        return self.decoder(x)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return (
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = VAE(
    latent_dim=LATENT_DIM
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded model:", MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

dataset = OASISDataset(
    DATA_DIR
)

print(
    "Images available:",
    len(dataset)
)


# ============================================================
# 1. ORIGINAL VS RECONSTRUCTED IMAGES
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)

images = next(
    iter(loader)
).to(device)

with torch.no_grad():

    reconstructions, mu, logvar = model(
        images
    )

images = images.cpu().numpy()

reconstructions = (
    reconstructions
    .cpu()
    .numpy()
)


fig, axes = plt.subplots(
    2,
    8,
    figsize=(16, 4)
)

for i in range(8):

    axes[0, i].imshow(
        images[i, 0],
        cmap="gray"
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstructions[i, 0],
        cmap="gray"
    )

    axes[1, i].axis("off")


axes[0, 0].set_ylabel(
    "Original"
)

axes[1, 0].set_ylabel(
    "Reconstructed"
)

plt.tight_layout()

reconstruction_path = os.path.join(
    OUTPUT_DIR,
    "vae_reconstructions.png"
)

plt.savefig(
    reconstruction_path,
    dpi=150
)

plt.close()

print(
    "Saved:",
    reconstruction_path
)


# ============================================================
# 2. 2D LATENT SPACE
# ============================================================

latent_values = []

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)

with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        mu, logvar = model.encode(
            batch
        )

        latent_values.append(
            mu.cpu().numpy()
        )


latent_values = np.concatenate(
    latent_values,
    axis=0
)


plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    latent_values[:, 0],
    latent_values[:, 1],
    s=3,
    alpha=0.5
)

plt.xlabel(
    "Latent dimension 1"
)

plt.ylabel(
    "Latent dimension 2"
)

plt.title(
    "OASIS VAE 2D Latent Space"
)

plt.tight_layout()

latent_path = os.path.join(
    OUTPUT_DIR,
    "vae_latent_space.png"
)

plt.savefig(
    latent_path,
    dpi=150
)

plt.close()

print(
    "Saved:",
    latent_path
)


# ============================================================
# 3. SAMPLE NEW BRAINS FROM LATENT SPACE
# ============================================================

grid_size = 8

z_values = torch.linspace(
    -2.5,
    2.5,
    grid_size
)

z_grid = []

for y in reversed(z_values):

    for x in z_values:

        z_grid.append(
            [
                x.item(),
                y.item()
            ]
        )


z_grid = torch.tensor(
    z_grid,
    dtype=torch.float32,
    device=device
)


with torch.no_grad():

    generated = model.decode(
        z_grid
    )

generated = (
    generated
    .cpu()
    .numpy()
)


fig, axes = plt.subplots(
    grid_size,
    grid_size,
    figsize=(10, 10)
)

for i, ax in enumerate(
    axes.flat
):

    ax.imshow(
        generated[i, 0],
        cmap="gray"
    )

    ax.axis("off")


plt.suptitle(
    "OASIS VAE Samples Across 2D Latent Space"
)

plt.tight_layout()

samples_path = os.path.join(
    OUTPUT_DIR,
    "vae_latent_samples.png"
)

plt.savefig(
    samples_path,
    dpi=150
)

plt.close()

print(
    "Saved:",
    samples_path
)


print(
    "\nVAE visualisation complete."
)
