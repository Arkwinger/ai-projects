# train_dp.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from safetensors.torch import save_file

# =========================
# Configuration
# =========================

RANDOM_SEED = 1337

BATCH_SIZE = 256
EPOCHS = 20

LEARNING_RATE = 0.05

TARGET_EPSILON = 10.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)

# =========================
# SVHN Normalization
# =========================

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

# =========================
# Load Dataset
# =========================

print("Loading SVHN dataset...")

train_dataset = datasets.SVHN(
    root="data",
    split="train",
    download=True,
    transform=transform
)

test_dataset = datasets.SVHN(
    root="data",
    split="test",
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# CNN Architecture
# =========================

class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# =========================
# Initialize Model
# =========================

model = SVHNCNN().to(DEVICE)

model = ModuleValidator.fix(model)

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)

criterion = nn.CrossEntropyLoss()

# =========================
# Differential Privacy
# =========================

privacy_engine = PrivacyEngine(accountant="rdp")

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=TARGET_EPSILON,
    target_delta=DELTA,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"\nTraining with DP-SGD")
print(f"Target epsilon: {TARGET_EPSILON}")
print(f"Delta: {DELTA}")
print(f"Max grad norm: {MAX_GRAD_NORM}")

# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total

    epsilon = privacy_engine.get_epsilon(DELTA)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {running_loss:.2f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Epsilon: {epsilon:.2f}"
    )

# =========================
# Evaluation
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * correct / total

print(f"\nTest Accuracy: {test_acc:.2f}%")

# =========================
# Save Model
# =========================

save_file(model._module.state_dict(), "dp_model.safetensors")

print("\nSaved dp_model.safetensors")
```
