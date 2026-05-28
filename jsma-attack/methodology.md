# Methodology

## Objective

The goal of this project was to implement and analyze the Jacobian-based Saliency Map Attack (JSMA) against an MNIST image classifier under strict sparse perturbation constraints.

The attack focused on generating targeted adversarial examples by modifying only a limited number of strategically selected pixels while forcing the classifier to predict a chosen target class.

---

## Environment Setup

A dedicated Python virtual environment was created to isolate dependencies required for the attack implementation.

Required libraries included:

* PyTorch
* NumPy
* Requests
* Pillow
* Torchvision

The pretrained classifier weights were retrieved directly from the challenge server and loaded locally to enable gradient computation and saliency analysis.

---

## Model Reconstruction

The server-side classifier architecture was recreated locally using PyTorch.

The model followed a LeNet-5 style convolutional neural network architecture consisting of:

* Convolutional layers
* Average pooling layers
* Fully connected layers
* Tanh activation functions
* Log-softmax output layer

Recreating the model locally allowed direct computation of Jacobian gradients required for JSMA.

---

## Challenge Acquisition

The challenge image and attack parameters were retrieved from the API endpoint.

The challenge provided:

* Original image
* Ground-truth label
* Target misclassification class
* Maximum allowed L0 perturbation budget
* Maximum L2 constraint

The image was decoded from a base64 PNG representation into normalized pixel space.

---

## Jacobian Gradient Computation

Gradients were computed with respect to the target class output.

The attack calculated which pixels most strongly increased the target class score while minimizing influence on competing classes.

This produced a saliency ranking used to prioritize pixel modifications.

---

## Sparse Pixel Perturbation

Pixels were iteratively modified according to saliency ranking.

For each iteration:

1. Gradients were recomputed
2. The most influential unmodified pixel was selected
3. The pixel value was adjusted
4. The classifier prediction was reevaluated

The process continued until either:

* The classifier predicted the target class
* The perturbation budget was exhausted

All pixel values were constrained to remain within the valid range of [0,1].

---

## Constraint Validation

The attack monitored:

* L0 norm (number of modified pixels)
* L2 distance between original and adversarial image

This ensured the generated adversarial example satisfied server-side validation requirements.

---

## Submission and Verification

The final adversarial image was encoded back into PNG format and submitted to the challenge API.

Successful submissions required:

* Prediction equal to target class
* L0 perturbation count within allowed budget
* L2 distance below threshold
* Valid image formatting

Upon successful validation, the server returned the challenge flag.

---

## Research Focus

This project explored practical adversarial machine learning concepts including:

* Sparse adversarial perturbations
* Saliency-based feature selection
* Jacobian analysis
* Targeted adversarial attacks
* Neural network sensitivity to feature manipulation

The implementation demonstrated how small, highly strategic modifications can significantly alter neural network predictions while maintaining minimal overall visual distortion.
