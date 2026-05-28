# Methodology

## Objective

The objective of this research was to evaluate sparse adversarial attack techniques against a ResNet-18 image classification model trained on the CIFAR-10 dataset.

The project focused on generating targeted adversarial examples capable of forcing controlled misclassification while preserving relatively low visible distortion and satisfying server-side perturbation validation constraints.

The assessment specifically examined the effectiveness of Jacobian-based Saliency Map Attack (JSMA) methodologies against modern convolutional neural network architectures operating on RGB image data.

---

## Environment Preparation

A dedicated Python virtual environment was created to isolate dependencies and ensure reproducibility throughout testing.

The implementation utilized:

* Python
* PyTorch
* NumPy
* Pillow
* Requests
* Matplotlib

The challenge environment provided access to:

* Challenge samples
* Model metadata
* ResNet-18 model weights
* Validation endpoints
* Submission APIs

---

## Target Model Reconstruction

The target architecture was reconstructed locally using a ResNet-18 style convolutional neural network adapted for CIFAR-10 classification.

The architecture consisted of:

* Residual BasicBlock modules
* Batch normalization layers
* Adaptive average pooling
* Fully connected classification layers
* Residual shortcut connections

Unlike simpler MNIST classifiers used in earlier sparsity attack exercises, this model operated on RGB image tensors with significantly deeper feature extraction layers and residual learning behavior.

Local reconstruction enabled direct access to:

* Forward propagation
* Backpropagation
* Gradient computation
* Feature saliency analysis

---

## Challenge Acquisition

Challenge data was retrieved through the provided API endpoints.

Each challenge item supplied:

* Sample identifier
* Original classification label
* Target misclassification label
* Required attack method
* Base64-encoded RGB image

Images were decoded into normalized tensor space with dimensions:

```text
(1, 3, 32, 32)
```

representing RGB CIFAR-10 samples.

---

## CIFAR-10 Normalization

Input images were normalized according to CIFAR-10 dataset statistics prior to inference.

Normalization constants included:

* Mean:

  * 0.4914
  * 0.4822
  * 0.4465

* Standard deviation:

  * 0.2470
  * 0.2435
  * 0.2616

This preprocessing step ensured compatibility with the pretrained ResNet model.

---

## Jacobian-Based Saliency Analysis

The implementation utilized Jacobian-based saliency analysis to determine which image features most strongly influenced the target class prediction.

Gradients were computed with respect to the target class output, allowing the attack to identify highly influential pixels capable of increasing target-class activation.

The attack iteratively:

1. Computed target gradients
2. Ranked pixels by saliency magnitude
3. Selected high-impact perturbation candidates
4. Applied directional perturbations
5. Reevaluated classifier predictions

Unlike dense perturbation attacks, the methodology emphasized sparse feature manipulation through selective pixel modification.

---

## Sparse Perturbation Strategy

The adversarial perturbation process operated under constrained modification budgets.

The attack limited:

* Number of modified pixels
* Maximum perturbation magnitude
* Overall image-space distortion

Perturbations were applied using gradient-direction updates:

x_{adv}=x+\theta\cdot sign(\nabla_x J(x,t))

where:

* (x) represents the original image
* (t) represents the target class
* (\theta) controls perturbation magnitude
* (\nabla_x J(x,t)) represents the target gradient

This iterative process continued until:

* The classifier predicted the target class
* The perturbation budget was exhausted
* Maximum iteration count was reached

---

## PNG Round-Trip Validation

Before submission, adversarial images underwent PNG encoding and decoding to replicate the evaluator’s internal processing pipeline.

This step ensured perturbations survived:

* Quantization
* PNG compression
* Image serialization

Without round-trip validation, perturbations could partially disappear during image conversion and invalidate the attack.

---

## Constraint Enforcement

The evaluator enforced several anti-cheating validation mechanisms including:

* Targeted misclassification verification
* Attack signature validation
* Minimum perturbation thresholds
* Image integrity checks

A minimum L2 perturbation threshold prevented submission of near-identical clean images while still requiring controlled perturbation behavior.

The implementation therefore balanced:

* Attack strength
* Sparsity
* Perturbation realism
* Validation compliance

---

## Submission Workflow

Final adversarial examples were submitted to the validation endpoint using:

* Sample identifier
* Method label
* Base64-encoded PNG image

Successful validation required:

* Correct target misclassification
* Valid perturbation signature
* PNG compatibility
* Compliance with perturbation thresholds

Upon successful validation, the server returned the challenge flag.

---

## Research Focus

This project explored several adversarial machine learning concepts including:

* Sparse adversarial perturbations
* Gradient-guided feature manipulation
* Jacobian saliency analysis
* Neural network evasion
* Adversarial robustness limitations
* RGB image perturbation strategies
* Deep convolutional network sensitivity

The implementation demonstrated how carefully selected perturbations can alter high-confidence predictions in modern residual neural networks while preserving relatively low visual distortion.
