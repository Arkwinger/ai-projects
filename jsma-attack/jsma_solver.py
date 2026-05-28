#!/usr/bin/env python3

import os
import io
import base64
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# -----------------------------
# Image Helpers
# -----------------------------

def x01_from_b64_png(b64):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)


def b64_png_from_x01(x2d):
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.b64encode(buf.getvalue()).decode()


def count_modified_pixels(a, b, threshold=1e-6):
    return int(np.sum(np.abs(a - b) > threshold))


# -----------------------------
# Model
# -----------------------------

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)

        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


def mnist_normalize(x):
    return (x - MNIST_MEAN) / MNIST_STD


# -----------------------------
# Fetch Challenge
# -----------------------------

print("[*] Fetching challenge...")

challenge = requests.get(
    f"{BASE_URL}/challenge",
    timeout=10
).json()

x = x01_from_b64_png(challenge["image_b64"])

original_label = int(challenge["original_label"])
target_class = int(challenge["target_class"])
l0_budget = int(challenge["l0_budget"])
max_l2 = float(challenge["max_l2"])

print(f"[+] Original label: {original_label}")
print(f"[+] Target class:   {target_class}")
print(f"[+] L0 budget:      {l0_budget}")
print(f"[+] Max L2:         {max_l2}")


# -----------------------------
# Load Model
# -----------------------------

print("[*] Loading model...")

model = MNISTClassifier().eval()

state = torch.load(
    "jsma_weights.pth",
    map_location=torch.device("cpu")
)

model.load_state_dict(state)

print("[+] Model loaded")


# -----------------------------
# Prepare Tensor
# -----------------------------

x_tensor = torch.from_numpy(
    x[None, None, ...]
).float()

with torch.no_grad():
    clean_logits = model(mnist_normalize(x_tensor))
    clean_pred = int(torch.argmax(clean_logits, dim=1).item())

print(f"[+] Clean prediction: {clean_pred}")


# -----------------------------
# JSMA Attack
# -----------------------------

print("[*] Running JSMA attack...")

x_adv = x_tensor.clone().detach()
modified_pixels = set()

success = False

for step in range(l0_budget):

    x_adv.requires_grad_(True)

    output = model(mnist_normalize(x_adv))

    pred = int(torch.argmax(output, dim=1).item())

    if pred == target_class:
        print(f"[+] Target achieved at step {step}")
        success = True
        break

    target_score = output[0, target_class]

    model.zero_grad()

    if x_adv.grad is not None:
        x_adv.grad.zero_()

    target_score.backward()

    grad = x_adv.grad.detach().clone()

    grad_flat = grad.view(-1)

    sorted_indices = torch.argsort(
        grad_flat,
        descending=True
    )

    selected_idx = None

    for idx in sorted_indices:

        idx_int = int(idx.item())

        if idx_int not in modified_pixels:
            selected_idx = idx_int
            modified_pixels.add(idx_int)
            break

    if selected_idx is None:
        print("[-] No selectable pixels left")
        break

    with torch.no_grad():

        flat = x_adv.view(-1)

        current_val = flat[selected_idx].item()

        if current_val < 0.5:
            flat[selected_idx] = 1.0
        else:
            flat[selected_idx] = 0.0

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if step % 5 == 0:

        with torch.no_grad():
            current_pred = int(
                torch.argmax(
                    model(mnist_normalize(x_adv)),
                    dim=1
                ).item()
            )

        print(
            f"[Step {step:02d}] "
            f"Prediction: {current_pred} | "
            f"Pixels modified: {len(modified_pixels)}"
        )


# -----------------------------
# Final Evaluation
# -----------------------------

with torch.no_grad():

    final_logits = model(mnist_normalize(x_adv))

    final_pred = int(
        torch.argmax(final_logits, dim=1).item()
    )

print("\n==============================")
print(f"Final prediction: {final_pred}")
print(f"Target class:     {target_class}")
print(f"Pixels modified:  {len(modified_pixels)}")

l2 = float(
    torch.norm(
        (x_adv - x_tensor).view(-1),
        p=2
    ).item()
)

print(f"L2 distance:      {l2:.4f}")
print("==============================\n")


# -----------------------------
# Submit
# -----------------------------

print("[*] Submitting adversarial image...")

adv_np = x_adv.detach().numpy()[0, 0]

b64_img = b64_png_from_x01(adv_np)

response = requests.post(
    f"{BASE_URL}/submit",
    json={"image_b64": b64_img},
    timeout=30
)

print(f"[+] HTTP {response.status_code}")
print(response.text)
