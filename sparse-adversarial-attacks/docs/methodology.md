# Methodology

## Overview

This repository examines sparse adversarial evasion techniques against convolutional neural network classifiers using the ElasticNet Adversarial Attack (EAD) framework with FISTA optimization.

The objective was to generate adversarial examples capable of forcing model misclassification while remaining within constrained perturbation budgets involving Elastic-Net, L1, and L2 distance limits.

The attack targeted an MNIST convolutional neural network classifier and used locally computed gradients to iteratively optimize sparse perturbations.

---

# Attack Workflow

The attack process followed these stages:

1. Retrieve the baseline image and perturbation constraints from the remote API
2. Download and load the pretrained CNN classifier weights
3. Replicate the classifier architecture locally using PyTorch
4. Compute adversarial gradients against the local model
5. Optimize perturbations using ElasticNet regularization and FISTA optimization
6. Enforce perturbation sparsity through soft-thresholding
7. Perform binary search over the adversarial tradeoff constant
8. Generate constrained adversarial examples
9. Submit the final adversarial image for remote validation

---

# ElasticNet Optimization

The attack objective combined adversarial loss with sparse regularization:

```math
||x_{adv} - x||_2 + \beta ||x_{adv} - x||_1
```

Where:

* L1 regularization promotes sparsity
* L2 regularization constrains perturbation magnitude
* β controls the sparsity tradeoff

Unlike dense perturbation attacks that distribute low-amplitude noise across the entire image, ElasticNet optimization concentrates perturbations on a smaller number of high-impact features.

---

# FISTA Optimization

The implementation used the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) to optimize adversarial perturbations.

FISTA combines:

* gradient descent
* momentum acceleration
* proximal optimization
* soft-thresholding

to iteratively refine perturbations while enforcing sparsity constraints.

Momentum acceleration improved convergence speed during optimization, while soft-thresholding removed low-impact perturbation values that did not meaningfully contribute to adversarial success.

---

# Sparse Perturbation Behavior

Observed perturbations were concentrated primarily around:

* digit edges
* stroke boundaries
* curved feature regions
* high-gradient areas

rather than uniformly across the image.

This behavior demonstrated that the classifier relied heavily on localized structural features for prediction.

The perturbation distributions also highlighted how sparse optimization can manipulate neural network decision boundaries with minimal visual distortion.

---

# Constraint Enforcement

Adversarial examples were required to satisfy multiple simultaneous constraints:

* Elastic-Net distance bound
* L1 perturbation bound
* L2 perturbation bound
* valid pixel-space clipping within [0,1]

Binary search optimization was used to identify the minimum adversarial tradeoff constant capable of achieving successful misclassification while remaining inside all perturbation limits.

---

# Observations

The generated adversarial examples remained visually similar to the original handwritten digits while successfully altering classifier predictions.

Sparse perturbations consistently clustered around structurally important image regions rather than random background areas, demonstrating the effectiveness of gradient-guided sparse optimization techniques in adversarial machine learning.

The research also highlighted the sensitivity of convolutional neural networks to carefully optimized localized perturbations, even under restrictive perturbation budgets.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* PIL
* Requests

---

# Research Areas

* Adversarial Machine Learning
* Sparse Optimization
* Neural Network Evasion
* CNN Robustness Analysis
* Gradient-Based Optimization
* Decision Boundary Manipulation
* Feature Sensitivity Analysis
