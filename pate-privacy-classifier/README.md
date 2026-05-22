# PATE Privacy-Preserving Letter Classifier

Implemented a privacy-preserving handwritten letter classifier using the PATE (Private Aggregation of Teacher Ensembles) framework on the EMNIST Letters dataset.

The project demonstrates how differential privacy techniques can mitigate membership inference attacks while maintaining strong classification performance.

---

# Features

- PATE student-teacher architecture
- Differential privacy using noisy teacher aggregation
- Membership inference attack resistance
- EMNIST handwritten letter classification
- PyTorch implementation

---

# Technologies Used

- Python
- PyTorch
- torchvision
- scikit-learn
- safetensors
- NumPy

---

# Model Architecture

The classifier uses a Multi-Layer Perceptron (MLP):

```python
Hidden Layers: [256, 128]
Dropout: 0.2
Output Classes: 26 (A-Z)
```

---

# Privacy Design

The implementation works by:

1. Training multiple teacher models on disjoint private datasets
2. Aggregating teacher votes
3. Adding Laplace noise to preserve privacy
4. Training a student model exclusively on noisy labels

This prevents the deployed student model from directly memorizing sensitive training data.

---

# Dataset

Dataset used:
- EMNIST Letters

Preprocessing included:
- Pixel normalization
- StandardScaler normalization
- Label conversion from 1-26 → 0-25

---

# Results

| Metric | Result |
|---|---|
| Accuracy | >80% |
| MIA Advantage | <3% |

The final student model successfully satisfied both privacy and accuracy requirements.

---

# Skills Demonstrated

- Differential Privacy
- PATE Architecture
- Membership Inference Attack Mitigation
- Neural Network Training
- Privacy-Preserving Machine Learning
- PyTorch Development
- EMNIST Classification

---

# Example Validation Output

```json
{
  "accuracy": 0.80,
  "mia_advantage": 0.015,
  "passed": true
}
```
