import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DAWNBench - CIFAR-10 ResNet-18")
print("=" * 70)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if device.type != "cuda":
    raise RuntimeError("CUDA GPU is not available. This job must run on an A100 GPU.")

print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)


# ============================================================
# 2. CIFAR-10 data
# ============================================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# Larger batches make better use of the A100 GPU.
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Training batch size:", 512)


# ============================================================
# 3. ResNet-18
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        # CIFAR-10 uses 32x32 images, so use 3x3 stride 1
        # instead of the large ImageNet-style first convolution.
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(
            BasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


model = ResNet18(num_classes=10).to(device)

print("Model created.")
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# 4. Training configuration
# ============================================================

num_epochs = 1

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)

# Mixed precision for the A100.
scaler = torch.amp.GradScaler("cuda")

print()
print("Training configuration")
print("-" * 70)
print("Epochs:", num_epochs)
print("Optimizer: SGD")
print("Learning rate: 0.1")
print("Momentum: 0.9")
print("Weight decay: 5e-4")
print("Label smoothing: 0.1")
print("Learning-rate schedule: Cosine Annealing")
print("Mixed precision: FP16")
print("-" * 70)


# ============================================================
# 5. Training
# ============================================================

train_losses = []
train_accuracies = []
epoch_times = []

print()
print("Starting training...")
print("=" * 70)

total_start = time.perf_counter()

for epoch in range(num_epochs):

    epoch_start = time.perf_counter()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    epoch_time = time.perf_counter() - epoch_start

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    epoch_times.append(epoch_time)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_accuracy:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.2f}s",
        flush=True
    )

total_training_time = time.perf_counter() - total_start

print("=" * 70)
print("Training complete!")
print(f"Total training time: {total_training_time:.2f}s")
print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")


# ============================================================
# 6. Test evaluation
# ============================================================

print()
print("Evaluating on CIFAR-10 test set...")

model.eval()

test_correct = 0
test_total = 0

inference_start = time.perf_counter()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(images)

        predicted = outputs.argmax(dim=1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

inference_time = time.perf_counter() - inference_start

test_accuracy = 100.0 * test_correct / test_total

print("=" * 70)
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct: {test_correct}/{test_total}")
print(f"Inference time: {inference_time:.2f}s")
print(f"Training time: {total_training_time:.2f}s")
print("=" * 70)

print()
print("DAWNBench run finished successfully.")
