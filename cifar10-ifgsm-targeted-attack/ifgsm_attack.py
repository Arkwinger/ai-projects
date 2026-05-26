#!/usr/bin/env python3

import argparse
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    img.save(buffer, format="PNG")

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# =========================================================
# MODEL
# =========================================================

class CIFAR10CNN(nn.Module):

    def __init__(self, num_classes: int = 10):

        super(CIFAR10CNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2, 2)

        # FC Layers
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

def load_model(model_path: str, device: str = "cpu"):

    model = CIFAR10CNN(num_classes=10)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    return model


# =========================================================
# I-FGSM TARGETED ATTACK
# =========================================================

def ifgsm_targeted_attack(
    model,
    image,
    target_class,
    epsilon,
    mean,
    std,
    num_iterations=50,
    alpha=None,
    device="cpu",
):

    if alpha is None:
        alpha = epsilon / num_iterations

    mean_t = torch.tensor(
        mean,
        device=device
    ).view(3, 1, 1)

    std_t = torch.tensor(
        std,
        device=device
    ).view(3, 1, 1)

    x_adv = image.clone().to(device)

    x_orig = image.clone().to(device)

    target = torch.tensor(
        [target_class],
        device=device
    )

    print("\n" + "=" * 60)
    print("I-FGSM Targeted Attack")
    print("=" * 60)

    print(f"Target class: {target_class}")
    print(f"Epsilon: {epsilon:.6f}")
    print(f"Iterations: {num_iterations}")
    print(f"Alpha: {alpha:.6f}")

    print("=" * 60 + "\n")

    for iteration in range(num_iterations):

        x_norm = (
            x_adv - mean_t
        ) / std_t

        x_norm.requires_grad = True

        outputs = model(
            x_norm.unsqueeze(0)
        )

        loss = F.cross_entropy(
            outputs,
            target
        )

        model.zero_grad()

        loss.backward()

        grad_norm = x_norm.grad

        grad_pixel = grad_norm / std_t

        # TARGETED ATTACK
        x_adv = x_adv - alpha * grad_pixel.sign()

        # PROJECT TO L∞ BALL
        delta = x_adv - x_orig

        delta = torch.clamp(
            delta,
            -epsilon,
            epsilon
        )

        x_adv = x_orig + delta

        # CLIP TO VALID RANGE
        x_adv = torch.clamp(
            x_adv,
            0.0,
            1.0
        )

        x_adv = x_adv.detach()

        if (iteration + 1) % 10 == 0:

            with torch.no_grad():

                x_check = (
                    x_adv - mean_t
                ) / std_t

                pred = model(
                    x_check.unsqueeze(0)
                ).argmax(dim=1).item()

            print(
                f"Iteration {iteration + 1}/{num_iterations} "
                f"- Prediction: {pred}"
            )

    print("\nAttack Complete\n")

    return x_adv.detach().cpu()


# =========================================================
# MAIN SOLVER
# =========================================================

def solve_challenge(
    host: str,
    device: str = "cpu",
):

    print("\n" + "=" * 60)
    print("Skills Assessment 1")
    print("=" * 60 + "\n")

    # -----------------------------------------------------
    # DOWNLOAD MODEL
    # -----------------------------------------------------

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
    # LOAD MODEL
    # -----------------------------------------------------

    print("[+] Loading model...")

    model = load_model(
        weights_path,
        device=device
    )

    print(f"[+] Model loaded on {device}")

    # -----------------------------------------------------
    # FETCH CHALLENGE
    # -----------------------------------------------------

    print("\n[+] Fetching challenge...")

    response = requests.get(
        f"{host}/challenge"
    )

    challenge = response.json()

    print(
        f"Original: "
        f"{challenge['original_class_name']}"
    )

    print(
        f"Target: "
        f"{challenge['target_class_name']}"
    )

    print(
        f"Epsilon: "
        f"{challenge['epsilon']:.6f}"
    )

    # -----------------------------------------------------
    # DECODE IMAGE
    # -----------------------------------------------------

    image = base64_to_tensor(
        challenge["image"]
    )

    mean = challenge["normalization"]["mean"]

    std = challenge["normalization"]["std"]

    print(f"Image shape: {image.shape}")

    # -----------------------------------------------------
    # VERIFY CLEAN PRED
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

        clean_pred = model(
            img_norm.unsqueeze(0).to(device)
        ).argmax(dim=1).item()

    print(f"Clean prediction: {clean_pred}")

    # -----------------------------------------------------
    # RUN ATTACK
    # -----------------------------------------------------

    print("\n[+] Running attack...")

    adv_image = ifgsm_targeted_attack(
        model=model,
        image=image,
        target_class=challenge["target_class"],
        epsilon=challenge["epsilon"],
        mean=mean,
        std=std,
        num_iterations=50,
        device=device,
    )

    # -----------------------------------------------------
    # VERIFY ADVERSARIAL
    # -----------------------------------------------------

    with torch.no_grad():

        adv_norm = (
            adv_image - mean_t
        ) / std_t

        adv_pred = model(
            adv_norm.unsqueeze(0).to(device)
        ).argmax(dim=1).item()

    print(f"\nAdversarial prediction: {adv_pred}")

    # -----------------------------------------------------
    # SUBMIT
    # -----------------------------------------------------

    print("\n[+] Submitting...")

    adv_base64 = tensor_to_base64(
        adv_image
    )

    submit = requests.post(
        f"{host}/submit",
        json={"image": adv_base64},
    )

    result = submit.json()

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    validation = result["validation"]

    print(
        f"L∞ norm: "
        f"{validation['linf_norm']:.6f}"
    )

    print(
        f"L∞ satisfied: "
        f"{validation['linf_satisfied']}"
    )

    print(
        f"Valid range: "
        f"{validation['valid_range']}"
    )

    print(
        f"Adversarial class: "
        f"{validation['adversarial_class']}"
    )

    print(
        f"Target achieved: "
        f"{validation['target_achieved']}"
    )

    print(f"\nSuccess: {result['success']}")

    if result["success"]:

        print("\n" + "=" * 60)
        print(f"FLAG: {result['flag']}")
        print("=" * 60 + "\n")

    else:

        print(f"\n[-] {result['message']}")


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
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    solve_challenge(
        args.host,
        args.device,
    )


if __name__ == "__main__":
    main()
