# CIFAR-10 DeepFool Attack

Implementation of the DeepFool adversarial attack against a CIFAR-10 image classifier using PyTorch.

## Overview

This repository demonstrates how minimal adversarial perturbations can force a convolutional neural network to misclassify an image while remaining under a strict L2 constraint.

The implementation uses the DeepFool algorithm to iteratively estimate and cross the nearest decision boundary with the smallest possible perturbation.

Developed through hands-on adversarial machine learning and AI security research.

---

## Objectives

- Reconstruct the target CNN architecture locally
- Load server-provided model weights
- Generate minimal adversarial perturbations
- Maintain perturbations under a strict L2 threshold
- Force image misclassification with minimal visual distortion
- Understand decision-boundary based adversarial attacks

---

## Technologies Used

- Python 3
- PyTorch
- Torchvision
- NumPy
- Requests
- Pillow (PIL)

---

## Attack Methodology

The implementation follows the DeepFool algorithm:

1. Download the challenge image and model weights
2. Reconstruct the target CNN locally
3. Normalize the image using CIFAR-10 statistics
4. Compute gradients for all competing classes
5. Estimate the nearest decision boundary
6. Apply the minimal perturbation required to cross the boundary
7. Repeat until misclassification is achieved
8. Submit the adversarial example

Unlike FGSM-based attacks that use fixed gradient directions, DeepFool dynamically estimates the closest boundary at every iteration to minimize perturbation magnitude.

---

## Model Architecture

The target classifier is a convolutional neural network consisting of:

- 2 Convolutional layers
- Batch Normalization
- ReLU activations
- Max Pooling
- Dropout regularization
- 2 Fully Connected layers

The model classifies RGB CIFAR-10 images with shape:

```text
(3, 32, 32)
```

---

## Features

- DeepFool minimal perturbation attack
- Decision-boundary estimation
- L2-constrained optimization
- CIFAR-10 normalization handling
- CNN reconstruction
- Gradient-based optimization
- Automated API interaction
- PyTorch-based implementation

---

## Example Output

<img width="666" height="276" alt="image" src="https://github.com/user-attachments/assets/ba787357-d947-4350-bd87-433a923880bc" />


---

## Key Implementation Adjustments

- Used CPU-only PyTorch builds to avoid environment storage limitations
- Correctly converted gradients between normalized space and pixel space
- Applied perturbations within normalized feature space
- Added overshoot handling to ensure decision boundary crossing
- Verified perturbation magnitude against normalized-space L2 constraints
- Clipped adversarial images to valid pixel ranges `[0,1]`

---

## Repository Structure

```text
cifar10-deepfool-attack/
│
├── README.md
├── requirements.txt
├── deepfool_attack.py
├── screenshots/
│   └── successful_attack.png
└── output/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python3 deepfool_attack.py --host http://TARGET_IP:PORT
```
---

## Disclaimer

This repository was created for educational and research purposes within controlled lab environments. The techniques demonstrated are intended for defensive security research, adversarial robustness evaluation, and AI security education.
