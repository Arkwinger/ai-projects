# Sparse Adversarial Evasion Research Using ElasticNet and FISTA

## Abstract

This repository examines sparse adversarial evasion techniques against convolutional neural network classifiers using the ElasticNet Adversarial Attack (EAD) framework with FISTA optimization.

The research focuses on how constrained perturbations can manipulate neural network decision boundaries while minimizing perceptual distortion and maximizing sparsity. By combining L1 and L2 regularization, ElasticNet optimization produces adversarial examples that preserve visual similarity while successfully inducing model misclassification.

Experiments were conducted against an MNIST CNN classifier under strict perturbation constraints involving Elastic-Net, L1, and L2 distance bounds.

---

# Research Objectives

- Analyze sparse adversarial perturbation behavior
- Evaluate ElasticNet regularization in adversarial evasion
- Study gradient-driven feature sensitivity in CNN classifiers
- Examine perturbation concentration along discriminative image regions
- Explore optimization-driven decision boundary manipulation
- Investigate sparsity–distortion tradeoffs in adversarial machine learning

---

# Background

Adversarial machine learning demonstrates that deep neural networks can be manipulated through carefully optimized perturbations that remain nearly imperceptible to human observers.

Unlike dense perturbation methods such as FGSM or PGD, ElasticNet-based attacks emphasize sparse feature modification through combined L1 and L2 optimization. This produces perturbations concentrated on high-impact features rather than distributing noise uniformly across the input space.

The attack implemented in this repository uses:

- ElasticNet regularization
- FISTA optimization
- proximal gradient methods
- soft-thresholding
- momentum acceleration
- binary search optimization

to iteratively construct adversarial examples under constrained perturbation budgets.

---

# ElasticNet Formulation

The optimization objective combines adversarial loss with sparse regularization:

```math
||x_{adv} - x||_2 + \beta ||x_{adv} - x||_1
```

Where:

- L1 regularization promotes sparsity
- L2 regularization constrains perturbation energy
- β controls the sparsity tradeoff

This formulation encourages perturbations that modify fewer high-impact pixels rather than spreading low-amplitude noise across the entire image.

---

# FISTA Optimization

The implementation uses the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) to optimize perturbations efficiently.

FISTA combines:

- gradient descent
- momentum acceleration
- proximal optimization
- soft-thresholding

to iteratively refine sparse adversarial perturbations.

Soft-thresholding removes low-impact perturbation values during optimization, allowing the attack to preserve sparsity while maintaining adversarial effectiveness.

---

# Methodology

The attack workflow consists of:

1. Loading the pretrained MNIST CNN classifier
2. Fetching the baseline image and perturbation constraints
3. Computing adversarial gradients locally
4. Applying ElasticNet optimization through FISTA iterations
5. Performing binary search over the adversarial tradeoff constant
6. Generating constrained adversarial examples
7. Validating perturbation constraints against the remote classifier

Perturbations are constrained by:

- Elastic-Net distance
- L1 distance
- L2 distance
- valid pixel-space clipping in [0,1]

---

# Repository Structure

```text
elasticnet-adversarial-attack/
│
├── README.md
├── requirements.txt
├── elasticnet_solver.py
├── notebooks/
│   └── elasticnet_analysis.ipynb
└── docs/
    └── methodology.md
```

---

# Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

---

# Execution

Download classifier weights:

```bash
curl -s -o elasticnet_weights.pth "$BASE_URL/weights"
```

Execute ElasticNet optimization:

```bash
python3 elasticnet_solver.py --host "$BASE_URL" --weights elasticnet_weights.pth
```

---

# Example Results

```json
{
  "clean_pred": 9,
  "adv_pred": 8,
  "l1": 10.7388,
  "l2": 1.2979,
  "linf": 0.4794,
  "elastic": 1.4052
}
```

Observed perturbations concentrated primarily along:

- digit edges
- stroke boundaries
- high-gradient feature regions

demonstrating the effectiveness of sparse optimization in adversarial evasion.

---

# Research Areas

- Adversarial Machine Learning
- Sparse Optimization
- Neural Network Evasion
- CNN Robustness Analysis
- Decision Boundary Manipulation
- Gradient-Based Optimization
- Feature Sensitivity Analysis
- Adversarial Robustness

---

# Technologies

- Python
- PyTorch
- NumPy
- PIL
- Requests

---

# Disclaimer

This repository is intended exclusively for educational and adversarial machine learning research in controlled environments.

---

# References

- Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples
- Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- Adversarial Machine Learning research literature
- Hack The Box Academy
