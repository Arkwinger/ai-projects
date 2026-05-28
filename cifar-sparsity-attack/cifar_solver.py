#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass

import numpy as np
import urllib.request
from PIL import Image

import torch
import torch.nn as nn

# FORCE CPU
device = torch.device("cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SEED = 1337


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def cifar_normalize(x01: torch.Tensor):
    mean = torch.tensor(
        CIFAR10_MEAN,
        dtype=x01.dtype,
        device=x01.device
    )[None, :, None, None]

    std = torch.tensor(
        CIFAR10_STD,
        dtype=x01.dtype,
        device=x01.device
    )[None, :, None, None]

    return (x01 - mean) / std


def _to_b64_rgb_x01(x4d: np.ndarray):
    x = np.transpose(x4d[0], (1, 2, 0))

    x255 = np.clip(
        (x * 255.0).round(),
        0,
        255
    ).astype(np.uint8)

    img = Image.fromarray(x255, mode="RGB")

    buf = io.BytesIO()

    img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _x01_from_b64_rgb(b64: str):
    raw = base64.b64decode(b64)

    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x = np.asarray(img, dtype=np.float32) / 255.0

    x = np.transpose(x, (2, 0, 1))[None, ...]

    return x.astype(np.float32)


def _http_get_json(url: str):
    req = urllib.request.Request(url, method="GET")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict):
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# =========================================================
# ResNet
# =========================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):

        out = torch.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = torch.relu(out)

        return out


class ResNetCIFAR(nn.Module):

    def __init__(
        self,
        num_blocks=(2, 2, 2, 2),
        num_classes=10
    ):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):

        layers = []

        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        out = torch.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out


# =========================================================
# Challenge
# =========================================================

@dataclass
class ChallengeItem:
    sample_id: int
    label: int
    target: int
    required_method: str
    x01: np.ndarray


def fetch_challenge(host: str):

    payload = _http_get_json(f"{host}/challenge")

    out = []

    for it in payload["items"]:

        raw = base64.b64decode(it["image_b64"])

        img = Image.open(io.BytesIO(raw)).convert("RGB")

        x = np.asarray(img, dtype=np.float32) / 255.0

        x = np.transpose(x, (2, 0, 1))[None, ...]

        out.append(
            ChallengeItem(
                sample_id=int(it["sample_id"]),
                label=int(it["label"]),
                target=int(it["target"]),
                required_method=str(it["required_method"]).lower(),
                x01=x.astype(np.float32),
            )
        )

    return out


# =========================================================
# Load Model
# =========================================================

def load_model(weights: str):

    ckpt = torch.load(
        weights,
        map_location=device
    )

    model = ResNetCIFAR().to(device)

    state_dict = (
        ckpt.get("state_dict_ema")
        or ckpt.get("state_dict")
        or ckpt
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# =========================================================
# STRONGER JSMA
# =========================================================

def jsma_targeted(
    model,
    x01,
    target,
    theta=0.30,
    gamma=0.35,
    max_iter=800,
):

    x = x01.clone().detach()

    B, C, H, W = x.shape

    pixel_budget = max(1, int(gamma * H * W))

    modified = set()

    for step in range(max_iter):

        x_req = x.clone().detach().requires_grad_(True)

        logits = model(cifar_normalize(x_req))

        pred = int(torch.argmax(logits, dim=1).item())

        if pred == target:
            print(f"[+] Target achieved at step {step}")
            break

        target_score = logits[0, target]

        model.zero_grad()

        target_score.backward()

        grad = x_req.grad.detach()

        grad_pix = grad.abs().sum(dim=1).view(-1)

        sorted_idx = torch.argsort(
            grad_pix,
            descending=True
        )

        selected = None

        for idx in sorted_idx.tolist():

            if idx not in modified:
                selected = idx
                modified.add(idx)
                break

        if selected is None:
            break

        r = selected // W
        c = selected % W

        with torch.no_grad():

            direction = torch.sign(
                grad[0, :, r, c]
            )

            x[0, :, r, c] += theta * direction

            x = torch.clamp(x, 0.0, 1.0)

        if step % 10 == 0:

            with torch.no_grad():

                current_pred = int(
                    torch.argmax(
                        model(cifar_normalize(x)),
                        dim=1
                    ).item()
                )

            print(
                f"[Step {step}] "
                f"Prediction={current_pred} "
                f"Pixels={len(modified)}"
            )

        if len(modified) >= pixel_budget:
            break

    return x.detach()


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--weights",
        default="cifar10_model.pth"
    )

    args = parser.parse_args()

    set_seed(SEED)

    print("[*] Fetching challenge...")

    items = fetch_challenge(args.host)

    print("[*] Loading model...")

    model = load_model(args.weights)

    submissions = []

    for item in items:

        print(f"\n[Sample {item.sample_id}]")
        print(f"Original: {item.label}")
        print(f"Target:   {item.target}")

        x = torch.from_numpy(item.x01).to(device)

        adv = jsma_targeted(
            model,
            x,
            target=item.target
        )

        adv_np = adv.detach().cpu().numpy()

        b64 = _to_b64_rgb_x01(adv_np)

        adv_q = torch.from_numpy(
            _x01_from_b64_rgb(b64)
        ).to(device)

        with torch.no_grad():

            pred = int(
                torch.argmax(
                    model(cifar_normalize(adv_q)),
                    dim=1
                ).item()
            )

        print(f"Final prediction: {pred}")

        submissions.append(
            {
                "sample_id": item.sample_id,
                "method": "jacobian",
                "image_b64": b64,
            }
        )

    print("\n[*] Submitting...")

    resp = _http_post_json(
        f"{args.host}/submit_images",
        {"items": submissions}
    )

    print(json.dumps(resp, indent=2))

    if resp.get("flag"):
        print("\nFlag:", resp["flag"])


if __name__ == "__main__":
    main()
