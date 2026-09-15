import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/home/groups/comp3710/OASIS/keras_png_slices_train"
OUTPUT_DIR = "gan_results"

IMAGE_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_WORKERS = 8

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================

class OASISDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".png")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.data_dir, filename)

        image = Image.open(path).convert("L")
        image = self.transform(image)

        return image


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 100 -> 512 x 4 x 4
            nn.ConvTranspose2d(
                LATENT_DIM, 512, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(
                512, 256, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(
                256, 128, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(
                128, 64, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(
                64, 1, 4, 2, 1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 64 -> 32
            nn.utils.spectral_norm(
                nn.Conv2d(1, 64, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, 4, 2, 1)
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, 4, 2, 1)
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, 4, 2, 1)
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 -> 1
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1, 4, 1, 0)
            )
        )

    def forward(self, x):
        return self.model(x).view(-1)


# ============================================================
# Setup
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=== OASIS GAN ===")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

dataset = OASISDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

print("Training images:", len(dataset))
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# Models
# ============================================================

G = Generator().to(device)
D = Discriminator().to(device)

print(
    "Generator parameters:",
    sum(p.numel() for p in G.parameters())
)

print(
    "Discriminator parameters:",
    sum(p.numel() for p in D.parameters())
)


# ============================================================
# Loss and optimizers
# ============================================================

# Hinge GAN loss is generally more stable than vanilla BCE GAN.
optimizer_G = optim.Adam(
    G.parameters(),
    lr=LR,
    betas=(BETA1, BETA2)
)

optimizer_D = optim.Adam(
    D.parameters(),
    lr=LR,
    betas=(BETA1, BETA2)
)


def discriminator_loss(real_scores, fake_scores):
    real_loss = torch.relu(1.0 - real_scores).mean()
    fake_loss = torch.relu(1.0 + fake_scores).mean()
    return real_loss + fake_loss


def generator_loss(fake_scores):
    return -fake_scores.mean()


# ============================================================
# Fixed latent vectors
# ============================================================

fixed_noise = torch.randn(
    64,
    LATENT_DIM,
    1,
    1,
    device=device
)

history = []

start_time = time.time()


# ============================================================
# Training
# ============================================================

for epoch in range(1, EPOCHS + 1):

    G.train()
    D.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    batches = 0

    for real_images in loader:

        real_images = real_images.to(
            device,
            non_blocking=True
        )

        batch_size = real_images.size(0)

        # ----------------------------------------------------
        # Train Discriminator
        # ----------------------------------------------------

        optimizer_D.zero_grad(set_to_none=True)

        real_scores = D(real_images)

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=device
        )

        fake_images = G(noise)

        fake_scores = D(fake_images.detach())

        d_loss = discriminator_loss(
            real_scores,
            fake_scores
        )

        d_loss.backward()
        optimizer_D.step()

        # ----------------------------------------------------
        # Train Generator
        # ----------------------------------------------------

        optimizer_G.zero_grad(set_to_none=True)

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=device
        )

        generated = G(noise)

        generated_scores = D(generated)

        g_loss = generator_loss(
            generated_scores
        )

        g_loss.backward()
        optimizer_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        batches += 1

    avg_g = total_g_loss / batches
    avg_d = total_d_loss / batches

    history.append((epoch, avg_g, avg_d))

    # --------------------------------------------------------
    # Save fixed samples
    # --------------------------------------------------------

    G.eval()

    with torch.no_grad():
        samples = G(fixed_noise)

    save_image(
        samples,
        os.path.join(
            OUTPUT_DIR,
            f"samples_epoch_{epoch:03d}.png"
        ),
        nrow=8,
        normalize=True
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"D loss: {avg_d:.4f} | "
        f"G loss: {avg_g:.4f}"
    )


# ============================================================
# Save final model
# ============================================================

torch.save(
    {
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "latent_dim": LATENT_DIM,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "history": history
    },
    os.path.join(
        OUTPUT_DIR,
        "oasis_gan_final.pth"
    )
)


# ============================================================
# Generate final samples
# ============================================================

G.eval()

with torch.no_grad():

    final_noise = torch.randn(
        64,
        LATENT_DIM,
        1,
        1,
        device=device
    )

    final_samples = G(final_noise)

save_image(
    final_samples,
    os.path.join(
        OUTPUT_DIR,
        "gan_final_samples.png"
    ),
    nrow=8,
    normalize=True
)


elapsed = time.time() - start_time

print()
print("Training completed.")
print(f"Training time: {elapsed:.2f} seconds")
print(
    "Saved:",
    os.path.join(
        OUTPUT_DIR,
        "oasis_gan_final.pth"
    )
)
print(
    "Saved:",
    os.path.join(
        OUTPUT_DIR,
        "gan_final_samples.png"
    )
)
