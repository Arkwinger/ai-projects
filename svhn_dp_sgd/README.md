# Building a Privacy-Preserving Digit Classifier with DP-SGD

A hands-on project exploring differential privacy in machine learning using the SVHN (Street View House Numbers) dataset.

The goal of this project was to train a CNN image classifier that maintained strong accuracy while resisting membership inference attacks using Differentially Private Stochastic Gradient Descent (DP-SGD).

---

# Project Goal

Build a privacy-preserving image classifier that balances:

| Metric | Goal |
|---|---|
| Classification Accuracy | High |
| Membership Inference Leakage | Low |

Final Results:

| Metric | Result |
|---|---|
| Test Accuracy | 81.88% |
| MIA Advantage | 0.006 |
| Privacy Budget (ε) | 10.0 |

---

# Concepts Explored

- Differential Privacy (DP)
- DP-SGD
- Membership Inference Attacks (MIA)
- Overfitting and memorization
- Gradient clipping
- Gaussian noise injection
- Privacy vs utility tradeoff
- PyTorch + Opacus

---

# What is a Membership Inference Attack?

A membership inference attack attempts to determine whether a specific sample was used during model training.

Machine learning models often become more confident on training data than unseen data. Attackers can exploit these confidence differences to infer membership.

This becomes dangerous when models are trained on sensitive information such as:

- medical data
- financial records
- private user activity
- biometric information

---

# How Differential Privacy Helps

DP-SGD modifies the training process itself by:

1. Clipping gradients to limit how much a single sample can influence training
2. Adding Gaussian noise to parameter updates
3. Reducing memorization of individual samples

This decreases membership inference leakage while preserving usable model accuracy.

---

# Dataset

This project uses the SVHN (Street View House Numbers) dataset from torchvision.

The dataset contains labeled images of house numbers captured from real-world street imagery.

---

# Model Architecture

```python
class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
```

---

# Differential Privacy Configuration

```python
TARGET_EPSILON = 10.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
```

Privacy was implemented using the Opacus library.

---

# Installation

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install opacus==1.4.0 --no-deps
pip install opt_einsum --no-deps
pip install safetensors --no-deps
pip install packaging --no-deps

pip install numpy scipy tqdm requests
```

---

# Running the Project

Train the model:

```bash
python3 train_dp.py
```

The script will:

- download the SVHN dataset
- normalize the data
- train the CNN with DP-SGD
- evaluate model accuracy
- save the trained model

---

# Example Training Output

```text
Epoch 19/20 Loss: 378.92 Train Acc: 81.59% Epsilon: 9.80
Epoch 20/20 Loss: 388.17 Train Acc: 81.87% Epsilon: 10.00

Test Accuracy: 81.43%

Saved dp_model.safetensors
```

---

# Key Takeaways

## Overfitting Causes Privacy Leakage

Models tend to memorize training samples when overfitting occurs.

This creates behavioral differences between:
- training data (members)
- unseen data (non-members)

Membership inference attacks exploit this gap.

---

## Differential Privacy Reduces Memorization

DP-SGD reduces the influence of individual training samples through:
- gradient clipping
- noise injection

This weakens membership inference attacks.

---

## Privacy vs Utility Tradeoff

Stronger privacy usually reduces model accuracy.

Finding a practical balance between:
- usability
- security
- privacy

is one of the core challenges of privacy-preserving machine learning.

---

# Technologies Used

- Python
- PyTorch
- Opacus
- Safetensors
- SVHN Dataset

---

# Disclaimer

This project was created for educational and research purposes to explore differential privacy and privacy-preserving machine learning techniques.
