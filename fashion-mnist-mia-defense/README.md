# Fashion-MNIST Membership Inference Defense

Implemented a privacy-aware Fashion-MNIST classifier designed to resist Membership Inference Attacks (MIA) while maintaining strong classification accuracy.

This project demonstrates practical machine learning privacy defenses using regularization and confidence-smoothing techniques.

---

# Overview

Membership Inference Attacks attempt to determine whether specific samples were used during a model’s training process.

This implementation reduces model overconfidence and overfitting using several defensive techniques while preserving model utility.

The defended model successfully:
- Maintained greater than 70% test accuracy
- Reduced MIA advantage by over 40% compared to the vulnerable baseline

---

# Features

- Fashion-MNIST CNN classifier
- Membership Inference Attack mitigation
- Dropout regularization
- Label smoothing
- Weight decay regularization
- PyTorch implementation
- Privacy-preserving model training

---

# Technologies Used

- Python
- PyTorch
- torchvision
- safetensors

---

# Model Architecture

The project uses the required `FashionMNISTCNN` architecture:

```python
Conv2d(1 → 32)
Conv2d(32 → 64)
Conv2d(64 → 64)

MaxPool2d
ReLU

Linear(64*3*3 → 128)
Linear(128 → 10)
```

Additional privacy-focused defenses included:
- Dropout (0.4)
- Label smoothing
- Weight decay

---

# Privacy Defense Techniques

## Dropout

Randomly disables neurons during training to reduce memorization and overfitting.

## Label Smoothing

Prevents the model from becoming overly confident on training samples, reducing vulnerability to confidence-based MIA attacks.

## Weight Decay

Regularizes model weights to improve generalization and reduce leakage from memorized samples.

---

# Dataset

Dataset used:
- Fashion-MNIST

Normalization:

```python
Normalize((0.2860,), (0.3530,))
```

---

# Results

| Metric | Result |
|---|---|
| Test Accuracy | >70% |
| MIA Reduction | >40% improvement |
| Privacy Defense | Successful |

The final model successfully resisted the standardized Membership Inference Attack evaluation.

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Training Script

```bash
python fashion_defense.py
```

This will:
- Download the Fashion-MNIST dataset
- Train the defended CNN model
- Apply privacy-preserving regularization techniques
- Export the final model as:

```text
defended_model.safetensors
```

---

## Submit the Model

Set the challenge server URL:

```bash
export BASE_URL="http://<ip>:<port>"
```

Submit the trained model:

```bash
curl -s -X POST "$BASE_URL/submit" \
  -F "defended_model=@defended_model.safetensors" | jq
```

---

# Skills Demonstrated

- Machine Learning Privacy
- Membership Inference Attack Mitigation
- Privacy-Aware Neural Network Training
- CNN Development
- PyTorch
- Model Regularization
- AI Security
- Fashion-MNIST Classification

---

# Example Validation Output

```json
{
  "valid": true,
  "test_accuracy": 0.71,
  "mia_advantage": 0.01,
  "improvement_ratio": 0.90
}
```
