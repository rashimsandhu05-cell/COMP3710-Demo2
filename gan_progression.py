import os
import matplotlib.pyplot as plt
from PIL import Image

epochs = [1, 10, 20, 30, 40, 50]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, epoch in zip(axes.flat, epochs):
    path = f"gan_results/samples_epoch_{epoch:03d}.png"

    image = Image.open(path)

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Epoch {epoch}")
    ax.axis("off")

plt.tight_layout()

output = "gan_results/gan_training_progression.png"
plt.savefig(output, dpi=150)
plt.close()

print("Saved:", output)
