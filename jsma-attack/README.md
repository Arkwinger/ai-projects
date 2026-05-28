# JSMA Attack Research

Implementation and analysis of the Jacobian-based Saliency Map Attack (JSMA) against an MNIST image classifier.

## Overview

This project explores sparse adversarial machine learning attacks through targeted pixel manipulation using Jacobian saliency analysis.

The implementation focuses on generating adversarial examples capable of forcing controlled misclassification while operating under strict L0 perturbation constraints.

The attack workflow includes:

- Local reconstruction of the target classifier
- Jacobian gradient computation
- Saliency map generation
- Sparse adversarial perturbation
- Targeted misclassification under constrained budgets

---

## Repository Structure

```text
jsma-attack/
│
├── jsma_solver.py
├── methodology.md
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pillow
- Requests

---

## Research Topics

- Adversarial Machine Learning
- Jacobian-based Saliency Map Attack (JSMA)
- Sparse Adversarial Perturbations
- L0-Constrained Optimization
- Neural Network Evasion
- Gradient-Based Feature Manipulation

---

## Setup

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Download model weights:

```bash
curl -s -o jsma_weights.pth "$BASE_URL/weights"
```

Run the attack:

```bash
python3 jsma_solver.py
```

---

## Methodology

The attack reconstructs the target classifier locally in order to compute Jacobian gradients and saliency maps directly against the neural network.

Pixels are iteratively ranked according to their influence on the target class output. The attack modifies the most salient pixels while remaining within the allowed L0 perturbation budget.

The workflow includes:

1. Fetching the challenge image and attack parameters
2. Rebuilding the target model locally
3. Computing target-class gradients
4. Ranking pixels by saliency
5. Iteratively modifying selected pixels
6. Validating perturbation constraints
7. Submitting the adversarial example

---

## Research Focus

This project examines how sparse, highly targeted perturbations can significantly alter neural network predictions while minimizing total feature modification.

The implementation demonstrates practical adversarial attack development against image classification systems and explores:

- Feature sensitivity
- Gradient-based optimization
- Sparse perturbation generation
- Neural network decision boundary manipulation

---

## Notes

This repository was created for educational and adversarial AI security research purposes only.
