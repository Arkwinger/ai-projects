#!/usr/bin/env python3

import argparse
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import os


# =========================================================
# IMAGE HELPERS
# =========================================================

def base64_to_tensor(base64_str: str) -> torch.Tensor:

    img_bytes = base64.b64decode(base64_str)

    img = Image.open(io.BytesIO(img_bytes))

    tensor = transforms.ToTensor()(img)

    return tensor


def tensor_to_base64(tensor: torch.Tensor) -> str:

    img_array = (
        tensor.permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)

    img = Image.fromarray(img_array)

    buffer = io.BytesIO()

    img.save(buffer, format='PNG')

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode('utf-8')


# =========================================================
# MODEL
# =========================================================

class CIFAR10CNN(nn.Module):

    def __init__(self, num_classes: int = 10):

        super(CIFAR10CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            128
        )

        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x: torch.Tensor):

        x = self.pool1(
            self.relu1(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            self.relu2(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = x.view(x.size(0), -1)

        x = self.dropout(
            self.relu3(
                self.fc1(x)
            )
        )

        x = self.fc2(x)

        return x


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_path: str, device: str = 'cpu'):

    model = CIFAR10CNN(num_classes=10)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and 'model_state_dict' in checkpoint
    ):
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    return model


# =========================================================
# DEEPFOOL ATTACK
# =========================================================

def deepfool_attack(
    model: nn.Module,
    image: torch.Tensor,
    mean: list,
    std: list,
    num_classes: int = 10,
    overshoot: float = 0.02,
    max_iter: int = 50,
    device: str = 'cpu'
):

    mean_t = torch.tensor(
        mean,
        device=device
    ).view(3, 1, 1)

    std_t = torch.tensor(
        std,
        device=device
    ).view(3, 1, 1)

    x = image.clone().to(device)

    x_orig = image.clone().to(device)

    x_norm = (x - mean_t) / std_t

    with torch.no_grad():

        logits = model(
            x_norm.unsqueeze(0)
        )

        current_class = logits.argmax(
            dim=1
        ).item()

    original_class = current_class

    r_total = torch.zeros_like(x)

    print("\n" + "=" * 60)
    print("DeepFool Attack")
    print("=" * 60)

    print(f"Original class: {original_class}")
    print(f"Overshoot: {overshoot}")
    print(f"Max iterations: {max_iter}")

    print("=" * 60 + "\n")

    for iteration in range(max_iter):

        x_norm = (x - mean_t) / std_t

        x_norm.requires_grad = True

        logits = model(
            x_norm.unsqueeze(0)
        )

        current_class = logits.argmax(
            dim=1
        ).item()

        if current_class != original_class:

            print(
                f"[+] Misclassification achieved "
                f"at iteration {iteration + 1}"
            )

            print(f"[+] New class: {current_class}")

            break

        min_dist = float('inf')

        best_w = None

        best_f = None

        for k in range(num_classes):

            if k == current_class:
                continue

            if x_norm.grad is not None:
                x_norm.grad.zero_()

            logits[0, k].backward(
                retain_graph=True
            )

            grad_k = x_norm.grad.clone()

            if x_norm.grad is not None:
                x_norm.grad.zero_()

            logits[0, current_class].backward(
                retain_graph=True
            )

            grad_current = x_norm.grad.clone()

            w_k = grad_k - grad_current

            f_k = (
                logits[0, k]
                - logits[0, current_class]
            ).item()

            w_norm = torch.norm(w_k)

            dist = abs(f_k) / (
                w_norm + 1e-10
            )

            if dist < min_dist:

                min_dist = dist

                best_w = w_k

                best_f = f_k

        w_norm_sq = torch.norm(best_w) ** 2

        r_i = (
            abs(best_f)
            / (w_norm_sq + 1e-10)
        ) * best_w

        r_i_pixel = r_i * std_t

        r_total = r_total + (
            1 + overshoot
        ) * r_i_pixel

        x = x_orig + r_total

        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        if (iteration + 1) % 10 == 0:

            with torch.no_grad():

                x_norm_check = (
                    x - mean_t
                ) / std_t

                orig_norm = (
                    x_orig - mean_t
                ) / std_t

                l2 = torch.norm(
                    x_norm_check - orig_norm
                ).item()

            print(
                f"Iteration {iteration + 1}/{max_iter} "
                f"- L2 norm: {l2:.4f}"
            )

    print("\n" + "=" * 60)
    print("Attack Complete")
    print("=" * 60 + "\n")

    return (
        x.detach().cpu(),
        r_total.detach().cpu(),
        iteration + 1,
        current_class
    )


# =========================================================
# MAIN
# =========================================================

def solve_challenge(
    host: str,
    device: str = 'cpu'
):

    print("\n" + "=" * 60)
    print("Skills Assessment 2")
    print("=" * 60 + "\n")

    weights_path = "cifar10_model_best.pth"

    if not os.path.exists(weights_path):

        print("[+] Downloading model weights...")

        resp = requests.get(
            f"{host}/model/weights",
            timeout=30
        )

        with open(weights_path, "wb") as f:
            f.write(resp.content)

        print(f"[+] Saved to {weights_path}")

    # -----------------------------------------------------

    print("[+] Loading model...")

    model = load_model(
        weights_path,
        device=device
    )

    print(f"[+] Model loaded on {device}")

    # -----------------------------------------------------

    print("\n[+] Fetching challenge...")

    response = requests.get(
        f"{host}/challenge"
    )

    challenge = response.json()

    print(
        f"Original class: "
        f"{challenge['original_class_name']}"
    )

    print(
        f"L2 threshold: "
        f"{challenge['l2_threshold']}"
    )

    print(
        f"Overshoot hint: "
        f"{challenge['overshoot_hint']}"
    )

    # -----------------------------------------------------

    image = base64_to_tensor(
        challenge['image']
    )

    mean = challenge['normalization']['mean']

    std = challenge['normalization']['std']

    print(f"Image shape: {image.shape}")

    # -----------------------------------------------------

    with torch.no_grad():

        mean_t = torch.tensor(mean).view(
            3,
            1,
            1
        )

        std_t = torch.tensor(std).view(
            3,
            1,
            1
        )

        img_norm = (
            image - mean_t
        ) / std_t

        orig_pred = model(
            img_norm.unsqueeze(0).to(device)
        ).argmax(dim=1).item()

    print(f"Verified prediction: {orig_pred}")

    # -----------------------------------------------------

    print("\n[+] Running DeepFool attack...")

    adv_image, perturbation, iters, final_class = deepfool_attack(
        model=model,
        image=image,
        mean=mean,
        std=std,
        num_classes=challenge['num_classes_hint'],
        overshoot=challenge['overshoot_hint'],
        max_iter=challenge['max_iterations_hint'],
        device=device
    )

    # -----------------------------------------------------

    with torch.no_grad():

        mean_t = torch.tensor(mean).view(
            3,
            1,
            1
        )

        std_t = torch.tensor(std).view(
            3,
            1,
            1
        )

        orig_norm = (
            image - mean_t
        ) / std_t

        adv_norm = (
            adv_image - mean_t
        ) / std_t

        delta_norm = adv_norm - orig_norm

        l2_norm = torch.norm(
            delta_norm
        ).item()

    print(f"L2 norm: {l2_norm:.4f}")

    print(
        f"Threshold: "
        f"{challenge['l2_threshold']}"
    )

    print(
        f"Within threshold: "
        f"{l2_norm <= challenge['l2_threshold']}"
    )

    # -----------------------------------------------------

    print("\n[+] Submitting...")

    adv_base64 = tensor_to_base64(
        adv_image
    )

    response = requests.post(
        f"{host}/submit",
        json={"image": adv_base64}
    )

    result = response.json()

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    validation = result['validation']

    print(
        f"L2 norm: "
        f"{validation['l2_norm']:.4f} "
        f"/ "
        f"{validation['l2_threshold']}"
    )

    print(
        f"L2 satisfied: "
        f"{validation['l2_satisfied']}"
    )

    print(
        f"Valid range: "
        f"{validation['valid_range']}"
    )

    print(
        f"Original class: "
        f"{validation['original_class']}"
    )

    print(
        f"Adversarial class: "
        f"{validation['adversarial_class']}"
    )

    print(
        f"Misclassification: "
        f"{validation['misclassification']}"
    )

    print(f"\nSuccess: {result['success']}")

    if result['success']:

        print("\n" + "=" * 60)
        print(f"FLAG: {result['flag']}")
        print("=" * 60 + "\n")

    else:

        print(f"\n[-] {result['message']}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--host',
        required=True,
    )

    parser.add_argument(
        '--device',
        default=(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        ),
    )

    args = parser.parse_args()

    solve_challenge(
        args.host,
        args.device
    )


if __name__ == "__main__":
    main()
