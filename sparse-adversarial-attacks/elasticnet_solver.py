#!/usr/bin/env python3
from __future__ import annotations
import argparse
import base64
import io
import json
import time
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

import torch
import torch.nn as nn


def set_seed(seed: int = 1337) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SimpleClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = torch.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)

        return x


def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - MNIST_MEAN) / MNIST_STD


def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)

    img = Image.open(io.BytesIO(raw)).convert("L")

    x = np.asarray(img, dtype=np.float32) / 255.0

    return np.clip(x, 0.0, 1.0)


def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)

    img = Image.fromarray(x255, mode="L")

    buf = io.BytesIO()

    img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class Challenge:
    label: int
    beta: float
    elastic_max: float
    l2_max: float
    l1_max: float
    sample_index: int
    x01: np.ndarray


def fetch_challenge(host: str) -> Challenge:
    r = requests.get(f"{host}/challenge", timeout=10)

    r.raise_for_status()

    p = r.json()

    x2d = x01_from_b64_png(p["image_b64"])

    x4d = x2d[None, None, ...].astype(np.float32)

    return Challenge(
        label=int(p["label"]),
        beta=float(p["beta"]),
        elastic_max=float(p["elastic_max"]),
        l2_max=float(p["l2_max"]),
        l1_max=float(p["l1_max"]),
        sample_index=int(p["sample_index"]),
        x01=x4d,
    )


class EAD:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        beta: float,
        max_iter: int = 400,
        lr: float = 1e-2,
        bin_steps: int = 5,
        initial_const: float = 1e-3,
    ) -> None:

        self.model = model
        self.device = device

        self.beta = beta
        self.max_iter = max_iter
        self.lr = lr
        self.bin_steps = bin_steps
        self.initial_const = initial_const

        self.confidence = 0.0

        self.orig = None

    def _loss(self, x: torch.Tensor, y_onehot: torch.Tensor, const: torch.Tensor):

        logits = self.model(mnist_normalize(x))

        l1 = torch.sum(torch.abs(x - self.orig), dim=(1, 2, 3))

        l2 = torch.sum((x - self.orig) ** 2, dim=(1, 2, 3))

        elastic = l2 + self.beta * l1

        real = torch.sum(y_onehot * logits, dim=1)

        other = torch.max(
            (1 - y_onehot) * logits - y_onehot * 1e4,
            dim=1
        )[0]

        adv_loss = torch.clamp(real - other + self.confidence, min=0)

        total = (
            torch.sum(const * adv_loss)
            + torch.sum(l2)
            + self.beta * torch.sum(l1)
        )

        return total, l1, l2, elastic

    @torch.no_grad()
    def _prox(self, x: torch.Tensor, y: torch.Tensor, beta: float, step: int):

        zt = step / (step + 3.0)

        diff = y - self.orig

        cond1 = (diff > beta).float()
        cond2 = (torch.abs(diff) <= beta).float()
        cond3 = (diff < -beta).float()

        upper = torch.minimum(
            y - beta,
            torch.tensor(1.0, device=self.device)
        )

        lower = torch.maximum(
            y + beta,
            torch.tensor(0.0, device=self.device)
        )

        x_new = (
            cond1 * upper
            + cond2 * self.orig
            + cond3 * lower
        )

        y_new = x_new + zt * (x_new - x)

        return x_new, y_new

    def run(self, x01: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        bsz = x01.shape[0]

        self.orig = x01.clone().to(self.device)

        y_onehot = torch.zeros(bsz, 10, device=self.device)

        y_onehot.scatter_(1, y.view(-1, 1), 1)

        lower = torch.zeros(bsz, device=self.device)

        upper = torch.ones(bsz, device=self.device) * 1e10

        const = torch.ones(bsz, device=self.device) * self.initial_const

        best = x01.clone().to(self.device)

        best_dist = torch.ones(bsz, device=self.device) * 1e10

        for b in range(self.bin_steps):

            print(f"[+] Binary step {b+1}/{self.bin_steps}")

            x = x01.clone().to(self.device)

            yv = x01.clone().to(self.device)

            for it in range(self.max_iter):

                yv.requires_grad_(True)

                total, l1, l2, elastic = self._loss(
                    yv,
                    y_onehot,
                    const
                )

                grad = torch.autograd.grad(total, yv)[0]

                with torch.no_grad():

                    yv -= self.lr * grad

                    x, yv = self._prox(x, yv, self.beta, it + 1)

                x.clamp_(0.0, 1.0)

                yv.clamp_(0.0, 1.0)

                if it % 100 == 0:

                    with torch.no_grad():

                        preds = torch.argmax(
                            self.model(mnist_normalize(x)),
                            dim=1
                        )

                        for i in range(bsz):

                            if preds[i] != y[i] and elastic[i] < best_dist[i]:

                                best_dist[i] = elastic[i]

                                best[i] = x[i]

            with torch.no_grad():

                preds = torch.argmax(
                    self.model(mnist_normalize(x)),
                    dim=1
                )

                for i in range(bsz):

                    if preds[i] != y[i]:

                        upper[i] = min(upper[i], const[i])

                        const[i] = (
                            (lower[i] + upper[i]) / 2
                            if upper[i] < 1e9
                            else const[i]
                        )

                    else:

                        lower[i] = max(lower[i], const[i])

                        const[i] = (
                            (lower[i] + upper[i]) / 2
                            if upper[i] < 1e9
                            else const[i] * 10
                        )

        return best


def load_model(weights: str, device: torch.device) -> nn.Module:

    m = SimpleClassifier().to(device).eval()

    m.load_state_dict(torch.load(weights, map_location=device))

    return m


def _post_submit(host: str, b64: str) -> requests.Response:

    return requests.post(
        f"{host}/submit",
        json={"image_b64": b64},
        timeout=30
    )


def _post_submit_retry(
    host: str,
    b64: str,
    retries: int = 3,
    delay: float = 0.5
) -> requests.Response:

    last = None

    for _ in range(max(1, retries)):

        try:
            resp = _post_submit(host, b64)

            return resp

        except requests.RequestException as e:

            last = e

            time.sleep(delay)

    if last is not None:
        raise last


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--weights",
        default="elasticnet_weights.pth"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=400
    )

    parser.add_argument(
        "--bin-steps",
        type=int,
        default=5
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2
    )

    args = parser.parse_args()

    set_seed(1337)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    chall = fetch_challenge(args.host)

    print("[+] Challenge loaded")

    print(json.dumps({
        "label": chall.label,
        "beta": chall.beta,
        "elastic_max": chall.elastic_max,
        "l2_max": chall.l2_max,
        "l1_max": chall.l1_max,
    }, indent=2))

    model = load_model(args.weights, device)

    x = torch.from_numpy(chall.x01).to(device)

    y = torch.tensor(
        [chall.label],
        device=device,
        dtype=torch.long
    )

    clean_pred = int(
        torch.argmax(
            model(mnist_normalize(x)),
            dim=1
        ).item()
    )

    print(f"[+] Clean prediction: {clean_pred}")

    ead = EAD(
        model,
        device,
        beta=chall.beta,
        max_iter=args.max_iter,
        lr=args.lr,
        bin_steps=args.bin_steps,
    )

    adv = ead.run(x, y)

    with torch.no_grad():

        adv_pred = int(
            torch.argmax(
                model(mnist_normalize(adv)),
                dim=1
            ).item()
        )

    diff = adv.detach().cpu().numpy() - x.detach().cpu().numpy()

    l1 = float(np.sum(np.abs(diff)))

    l2 = float(np.sqrt(np.sum(diff**2)))

    linf = float(np.max(np.abs(diff)))

    elastic = l2 + chall.beta * l1

    print(json.dumps({
        "clean_pred": clean_pred,
        "adv_pred": adv_pred,
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "elastic": elastic,
        "beta": chall.beta,
    }, indent=2))

    b64 = b64_png_from_x01(
        adv.detach().cpu().numpy()[0, 0]
    )

    r = _post_submit_retry(args.host, b64)

    try:
        r.raise_for_status()

    except Exception:

        print("[-] Server response:")

        print(r.text)

        raise

    print("\n[+] FLAG:")

    print(r.json().get("flag"))


if __name__ == "__main__":
    main()
