#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import os

import numpy as np
import requests
from PIL import Image

import torch
import torch.nn as nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# =========================================================
# MODEL
# =========================================================

class SimpleClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01):

        x = (x01 - MNIST_MEAN) / MNIST_STD

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = torch.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))

        x = self.dropout2(x)

        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)


# =========================================================
# IMAGE HELPERS
# =========================================================

def x01_from_b64_png(b64):

    raw = base64.b64decode(b64)

    img = Image.open(io.BytesIO(raw)).convert("L")

    x = np.asarray(img, dtype=np.float32) / 255.0

    return np.clip(x, 0.0, 1.0)


def b64_png_from_x01(x2d):

    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)

    img = Image.fromarray(x255, mode="L")

    buf = io.BytesIO()

    img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def l2(a, b):

    return float(np.linalg.norm((a - b).ravel(), ord=2))


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(weights_path):

    model = SimpleClassifier()

    state = torch.load(weights_path, map_location=torch.device("cpu"))

    model.load_state_dict(state)

    model.eval()

    return model


# =========================================================
# DEEPFOOL-STYLE TARGETED ATTACK
# =========================================================

def targeted_attack(
    model,
    x01,
    target_label,
    l2_threshold,
    steps=1000,
    alpha=0.01,
):

    x_orig = torch.from_numpy(x01).float()

    x_adv = x_orig.clone().detach()

    best_adv = x_adv.clone().detach()

    best_conf = -1.0

    for i in range(steps):

        x_adv.requires_grad_(True)

        logits = model(x_adv)

        probs = torch.softmax(logits, dim=1)

        pred = int(torch.argmax(probs, dim=1).item())

        target_conf = float(probs[0, target_label].item())

        if target_conf > best_conf:
            best_conf = target_conf
            best_adv = x_adv.detach().clone()

        if pred == target_label:

            dist = torch.norm(x_adv - x_orig).item()

            print(f"[+] HIT TARGET at iter {i}")
            print(f"[+] confidence={target_conf:.4f}")
            print(f"[+] l2={dist:.4f}")

            return x_adv.detach().cpu().numpy()

        # maximize target class directly
        loss = -logits[0, target_label]

        model.zero_grad()

        loss.backward()

        grad = x_adv.grad.detach()

        # normalized gradients instead of sign
        grad_norm = torch.norm(grad)

        if grad_norm != 0:
            grad = grad / grad_norm

        # targeted update
        x_adv = x_adv - alpha * grad

        # L2 projection
        delta = x_adv - x_orig

        delta_norm = torch.norm(delta)

        if delta_norm > l2_threshold:
            delta = delta * (l2_threshold / delta_norm)

        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

        if i % 50 == 0:

            dist = torch.norm(x_adv - x_orig).item()

            print(
                f"[{i}] "
                f"pred={pred} "
                f"target_conf={target_conf:.4f} "
                f"l2={dist:.4f}"
            )

    print("[*] Returning best candidate found")

    return best_adv.cpu().numpy()


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        required=True,
    )

    parser.add_argument(
        "--weights",
        default="deepfool_weights.pth",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # FETCH CHALLENGE
    # -----------------------------------------------------

    print("[+] Fetching challenge...")

    r = requests.get(f"{args.host}/challenge", timeout=10)

    payload = r.json()

    x2d = x01_from_b64_png(payload["image_b64"])

    x4d = x2d[None, None, ...]

    label = int(payload["label"])

    target = int(payload["target"])

    l2_threshold = float(payload["l2_threshold"])

    print(f"[+] Original Label : {label}")
    print(f"[+] Target Label   : {target}")
    print(f"[+] L2 Threshold   : {l2_threshold}")

    # -----------------------------------------------------
    # DOWNLOAD WEIGHTS
    # -----------------------------------------------------

    if not os.path.exists(args.weights):

        print("[+] Downloading weights...")

        wb = requests.get(f"{args.host}/weights", timeout=15).content

        with open(args.weights, "wb") as f:
            f.write(wb)

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("[+] Loading model...")

    model = load_model(args.weights)

    clean_tensor = torch.from_numpy(x4d).float()

    clean_logits = model(clean_tensor)

    clean_pred = int(torch.argmax(clean_logits, dim=1).item())

    print(f"[+] Clean prediction: {clean_pred}")

    # -----------------------------------------------------
    # RUN ATTACK
    # -----------------------------------------------------

    print("[+] Running targeted attack...")

    x_adv = targeted_attack(
        model=model,
        x01=x4d,
        target_label=target,
        l2_threshold=l2_threshold,
        steps=1000,
        alpha=0.01,
    )

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    adv_tensor = torch.from_numpy(x_adv).float()

    adv_logits = model(adv_tensor)

    adv_probs = torch.softmax(adv_logits, dim=1)

    adv_pred = int(torch.argmax(adv_probs, dim=1).item())

    adv_conf = float(adv_probs[0, adv_pred].item())

    dist = l2(x_adv, x4d)

    print("\n[+] FINAL RESULTS")
    print(f"Prediction : {adv_pred}")
    print(f"Confidence : {adv_conf:.4f}")
    print(f"L2 Distance: {dist:.4f}")

    if adv_pred != target:
        print("[-] Target class not reached")
        return

    if dist > l2_threshold:
        print("[-] L2 threshold exceeded")
        return

    # -----------------------------------------------------
    # SUBMIT
    # -----------------------------------------------------

    print("[+] Submitting...")

    x_submit = x_adv[0, 0]

    b64 = b64_png_from_x01(x_submit)

    submit = requests.post(
        f"{args.host}/submit",
        json={"image_b64": b64},
        timeout=10,
    )

    print("\n[+] SERVER RESPONSE")
    print(submit.text)

    try:
        print("\nFLAG:")
        print(submit.json()["flag"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
