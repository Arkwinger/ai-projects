# Clean Label Data Poisoning Attack (Machine Learning Adversarial Attack)

This project demonstrates a **Clean Label Attack**, a stealthy data poisoning technique where an attacker modifies **only the feature values** of specific training samples while leaving their **labels untouched**. Unlike traditional label flipping, clean-label poisoning is subtle, harder to detect, and highly targeted.

The goal of this project is to show how an attacker can intentionally shift the model’s decision boundaries so that a **chosen target instance** is misclassified at inference time—even though the target itself is never modified.

---

##  Attack Overview

A Clean Label Attack works by:

1. **Selecting a target sample** whose classification you want the model to misclassify.
2. **Finding several training samples from a different class** that lie close to this target in feature space.
3. **Slightly perturbing these training samples’ features** so they move across the decision boundary.
4. **Keeping their labels unchanged** to maintain the illusion of clean data.
5. **Retraining the model**, forcing its boundary to shift and causing the target sample to fall on the wrong side.

This shift causes the model to confidently misclassify the target sample during inference.

---

##  Technical Steps Implemented

- Generated a 3-class synthetic dataset using `make_blobs`
- Standardized features with `StandardScaler`
- Trained a baseline multi-class logistic regression classifier
- Identified a **target point** in Class 1
- Located the **nearest Class 0 neighbors** to that target using `NearestNeighbors`
- Computed a perturbation vector based on the model’s decision boundary:
  - Normal vector: `w0 - w1`
  - Push direction: `-(w0 - w1)`
  - Scaled using `epsilon_cross`
- Applied the perturbation to the selected neighbors while **keeping their labels as Class 0**
- Retrained model on the poisoned dataset
- Verified the shift in the decision boundary and the misclassification of the target sample
- Visualized clean vs. poisoned datasets

---

##  Visual Results

The poisoned dataset shows:

- **Target point (Class 1)** remains unchanged.
- **Perturbed neighbors (Class 0)** appear visually in the Class 1 region.
- This discrepancy forces the model to shift its Class 0–Class 1 boundary.

The result:  
 The target sample becomes misclassified as Class 0 after retraining.

---

##  Why Clean Label Attacks Matter

Clean Label Attacks are dangerous because:

- They do **not** require label tampering  
- They can bypass data validation pipelines  
- They leave minimal traces  
- They can target **specific individuals or samples**  
- They create highly controlled misclassifications

Real-world risks include:

- Misclassifying malware as benign
- Triggering false positives in fraud detection
- Evading biometric/face recognition systems
- Manipulating content moderation models

---

##  Technologies Used

- Python
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---


##  

Dominic D’Acri  
Cybersecurity & AI/ML Security Enthusiast  
Hack The Box Academy – Adversarial ML Module
