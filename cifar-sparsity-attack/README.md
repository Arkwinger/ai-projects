# CIFAR-10 Sparse Adversarial Attack Lab

Research and implementation of sparse adversarial perturbation techniques against a ResNet-based CIFAR-10 image classifier.

---

## Project Summary

This project demonstrates how carefully crafted perturbations can manipulate deep neural network predictions while maintaining relatively small visible image changes.

Using gradient-guided saliency analysis, the attack selectively modifies high-impact image regions in order to force targeted misclassification against a ResNet-18 classifier trained on CIFAR-10.

Unlike traditional noise-based attacks that perturb entire images, this implementation focuses on sparse feature manipulation and adversarial optimization under constrained perturbation budgets.

---

## Core Concepts Explored

- Sparse adversarial attacks
- Targeted neural network evasion
- Jacobian saliency analysis
- Gradient-based feature selection
- CIFAR-10 image classification
- ResNet-18 adversarial robustness
- Adversarial perturbation constraints

---

## Attack Workflow

The attack process included:

1. Retrieving challenge images and metadata from the API
2. Reconstructing the target ResNet architecture locally
3. Computing target-class gradients
4. Ranking image regions by saliency influence
5. Applying directional sparse perturbations
6. Validating perturbation survival after PNG encoding
7. Submitting adversarial examples for evaluation

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python |
| ML Framework | PyTorch |
| Image Processing | Pillow |
| Numerical Operations | NumPy |
| Networking | Requests |
| Visualization | Matplotlib |

---

## Repository Layout

```text
cifar-sparsity-attack/
│
├── cifar_solver.py
├── methodology.md
├── requirements.txt
├── README.md
└── screenshots/
    └── successful_attack.png
```

---

## Installation

Clone the repository and create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Download the model weights:

```bash
curl -sL "$BASE_URL/model/weights" -o cifar10_model.pth
```

Run the solver:

```bash
python3 cifar_solver.py --host "$BASE_URL" --weights cifar10_model.pth
```

---

## Technical Focus

The implementation primarily explored how neural networks respond to sparse, high-saliency perturbations rather than large-scale image distortion.

Special attention was given to:

- Feature sensitivity
- Gradient directionality
- Sparse perturbation efficiency
- Decision boundary manipulation
- Perturbation persistence after image serialization

The project also demonstrated how adversarial perturbations can survive PNG conversion and still maintain targeted misclassification behavior.

---

## Research Notes

This repository was created as part of practical adversarial AI security research and educational experimentation involving neural network robustness and evasion techniques.
