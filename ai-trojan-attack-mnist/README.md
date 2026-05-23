# AI Trojan Attack on MNIST CNN

## Overview

This project demonstrates a Trojan/Backdoor attack against a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The objective was to poison the model so that images of the digit `7` would be misclassified as the digit `1` whenever a white trigger pattern was placed in the bottom-left corner of the image.

The attack maintained high clean accuracy while achieving a high Attack Success Rate (ASR), demonstrating how machine learning systems can be manipulated through poisoned training data.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- JupyterLab
- MNIST Dataset

---

## Final Evaluation Results

![Final Results](images/final_results.png)

---

## Skills Demonstrated

- AI Security
- Adversarial Machine Learning
- Trojan/Backdoor Attacks
- Dataset Poisoning
- CNN Training
- PyTorch
- Machine Learning Evaluation
