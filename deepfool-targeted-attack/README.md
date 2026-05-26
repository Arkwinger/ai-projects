# DeepFool Targeted Attack

Targeted adversarial machine learning attack against an MNIST image classifier using a DeepFool-style iterative gradient optimization approach in PyTorch.

## Overview

This repository demonstrates how adversarial examples can be generated to force a neural network classifier into predicting a specific target class while remaining within a constrained perturbation budget.

The attack reconstructs the server-side model locally, computes gradients against the target class, and iteratively perturbs the input image while enforcing an L2 distance constraint.

Developed through hands-on adversarial machine learning and AI security research.

---

## Objectives

- Recreate the target neural network architecture locally
- Load server-provided model weights
- Generate targeted adversarial examples
- Maintain perturbations under a strict L2 threshold
- Successfully force misclassification into a chosen target class

---

## Technologies Used

- Python 3
- PyTorch
- NumPy
- Requests
- Pillow (PIL)

---

## Attack Methodology

The attack uses a DeepFool-style iterative optimization process:

1. Download the challenge image and model weights
2. Reconstruct the target CNN architecture locally
3. Compute gradients with respect to the target class
4. Iteratively update the image using normalized gradients
5. Project perturbations back into the allowed L2 constraint region
6. Submit the crafted adversarial example to the target API

Unlike standard FGSM attacks that use sign gradients, this implementation uses normalized raw gradients for more precise optimization under tight perturbation constraints.

---

## Model Architecture

The target classifier is a convolutional neural network consisting of:

- 2 Convolutional layers
- ReLU activations
- Max pooling
- Dropout regularization
- 2 Fully connected layers
- Log-softmax output layer

The model classifies 28x28 grayscale MNIST digit images.

---

## Features

- Targeted adversarial attack generation
- Iterative gradient optimization
- L2 constraint projection
- Local model reconstruction
- API-based attack workflow
- Automated challenge solving
- PyTorch-based implementation

---

## Example Output

```bash
[+] Original Label : 4
[+] Target Label   : 6
[+] L2 Threshold   : 0.75

[+] Running targeted attack...

[+] HIT TARGET at iter 312
[+] confidence=0.8124
[+] l2=0.7421

```

---

## Repository Structure

```text
deepfool-targeted-attack/
│
├── README.md
├── requirements.txt
├── student.py
├── screenshots/
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
python3 student.py --host http://TARGET_HOST:PORT
```

---

## Skills Demonstrated

- Adversarial Machine Learning
- AI Evasion Techniques
- Gradient-Based Optimization
- PyTorch Model Reconstruction
- Offensive AI Security
- API Interaction
- Neural Network Analysis
- Constraint-Based Optimization

---

## Disclaimer

This repository was created for educational and research purposes within controlled lab environments. The techniques demonstrated are intended for defensive security research, adversarial robustness evaluation, and AI security education.
```
