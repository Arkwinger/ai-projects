# Methodology

## Research Objective

The objective of this research was to examine the effectiveness of sparse adversarial perturbations against convolutional neural networks using the Jacobian-based Saliency Map Attack (JSMA). The project focused on generating targeted adversarial examples capable of forcing controlled misclassification while operating under strict L0 perturbation constraints.

Unlike broad perturbation attacks that distribute noise across an entire image, JSMA prioritizes highly selective feature manipulation by modifying only a limited number of strategically chosen pixels.

---

## Experimental Environment

A controlled Python environment was established to isolate all required dependencies and ensure reproducibility of the attack workflow.

The implementation utilized:

* Python
* PyTorch
* NumPy
* Pillow
* Requests

A virtual environment was used to maintain dependency consistency throughout testing and experimentation.

---

## Target Model Reconstruction

The target classifier architecture was reconstructed locally using a LeNet-5 inspired convolutional neural network architecture provided by the challenge environment.

The reconstructed model consisted of:

* Convolutional feature extraction layers
* Average pooling operations
* Fully connected classification layers
* Tanh activation functions
* Log-softmax output layer

Rebuilding the model locally enabled direct access to forward and backward propagation operations necessary for Jacobian gradient analysis.

---

## Challenge Acquisition and Data Preparation

Challenge samples were retrieved from the API interface provided by the environment.

Each challenge supplied:

* A baseline MNIST image
* Ground-truth classification label
* Required target class
* Maximum allowed L0 perturbation budget
* Maximum L2 constraint threshold

The supplied image was decoded from a base64 PNG representation and converted into normalized tensor space for gradient computation and adversarial manipulation.

---

## Jacobian Gradient Analysis

The attack relied on Jacobian-based saliency analysis to identify features most influential toward the target misclassification objective.

Gradients were computed with respect to the target class output, allowing the attack to measure how individual pixel perturbations influenced classifier confidence.

This process enabled construction of a saliency ranking that prioritized pixels capable of:

* Increasing target class activation
* Reducing competing class influence
* Maximizing adversarial effectiveness under sparse constraints

The resulting saliency map functioned as a feature importance distribution used to guide perturbation selection.

---

## Sparse Perturbation Strategy

Adversarial perturbations were introduced iteratively according to saliency ranking.

At each iteration:

1. The Jacobian matrix was recomputed
2. The most influential unmodified pixel was selected
3. Pixel intensity was modified within valid bounds
4. The classifier prediction was reevaluated

This iterative process continued until either:

* The classifier predicted the required target class
* The perturbation budget was exhausted

All perturbations were constrained to valid image-space values within the range [0,1].

---

## Constraint Enforcement

The implementation continuously monitored perturbation metrics throughout attack execution.

The primary constraint enforced was the L0 norm:

|x_{adv}-x|_0 \leq budget

This constrained the maximum number of modified pixels permitted during adversarial generation.

Additional monitoring of L2 distance ensured perturbations remained within challenge validation thresholds and prevented complete image replacement attacks.

---

## Adversarial Evaluation

Generated adversarial examples were evaluated against the locally reconstructed model prior to submission.

Evaluation metrics included:

* Predicted class
* Number of modified pixels
* L2 perturbation magnitude
* Overall perturbation sparsity

The adversarial image was then re-encoded into PNG format and submitted to the validation endpoint.

Successful attacks required:

* Exact target-class misclassification
* Compliance with L0 perturbation budget
* Compliance with L2 threshold restrictions
* Preservation of valid image formatting

---

## Research Findings

The research demonstrated that convolutional neural networks remain vulnerable to highly sparse, gradient-guided perturbations despite modifying only a limited subset of input features.

The implementation highlighted several important adversarial machine learning concepts:

* Neural network sensitivity to localized feature manipulation
* Effectiveness of Jacobian saliency analysis
* Sparse perturbation optimization
* Targeted adversarial misclassification
* Feature importance exploitation within deep learning systems

The project further reinforced how adversarial attacks can exploit nonlinear feature dependencies to alter model predictions while maintaining relatively low overall visual distortion.
