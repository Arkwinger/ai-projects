# CIFAR-10 I-FGSM Targeted Attack

Targeted adversarial machine learning attack against a CIFAR-10 image classifier using Iterative Fast Gradient Sign Method (I-FGSM) in PyTorch.

## Overview

This repository demonstrates how adversarial examples can be generated to force a convolutional neural network into predicting a specific target class while remaining inside a strict L∞ perturbation constraint.

The implementation performs a targeted attack that transforms a CIFAR-10 dog image into one classified as a cat by iteratively applying small gradient-based perturbations while maintaining visually minimal changes.

Developed through hands-on adversarial machine learning and AI security research.

---

## Objectives

- Reconstruct the target CNN architecture locally
- Load server-provided model weights
- Generate targeted adversarial examples
- Respect strict L∞ perturbation constraints
- Successfully force dog → cat misclassification
- Understand iterative gradient-based evasion attacks

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

The attack uses Iterative Fast Gradient Sign Method (I-FGSM):

1. Download the challenge image and model weights
2. Rebuild the target CNN locally
3. Normalize the image using CIFAR-10 statistics
4. Compute gradients toward the target class
5. Iteratively perturb the image using targeted FGSM updates
6. Project perturbations into the allowed L∞ region
7. Clip pixel values to valid image ranges
8. Submit the adversarial example

Unlike one-step FGSM attacks, I-FGSM performs many small iterative updates, allowing more precise optimization under tight perturbation constraints.

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

- Targeted adversarial attack generation
- Iterative FGSM implementation
- L∞ perturbation clipping
- CIFAR-10 normalization handling
- CNN reconstruction
- Automated API interaction
- PyTorch-based workflow
- Gradient-based optimization

---

## Example Output

<img width="924" height="1127" alt="succ3ssful_attack" src="https://github.com/user-attachments/assets/7d470f70-9576-4b6d-af05-f420f8ba487e" />

---

## Repository Structure

```text
cifar10-ifgsm-targeted-attack/
│
├── README.md
├── requirements.txt
├── targeted_ifgsm.py
├── screenshots/
│   └── succ3ssful_attack.png
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
python3 targeted_ifgsm.py --host http://TARGET_IP:PORT
```

---

## Disclaimer

This repository was created for educational and research purposes within controlled lab environments. The techniques demonstrated are intended for defensive security research, adversarial robustness evaluation, and AI security education.
