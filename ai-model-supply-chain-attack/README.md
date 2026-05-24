# PyTorch Model Deserialization RCE

## Overview

This project demonstrates how insecure deserialization within machine learning workflows can lead to arbitrary code execution through malicious PyTorch model artifacts.

The proof-of-concept explores the security risks associated with loading untrusted `.pth` files using `torch.load()`, highlighting how Python pickle deserialization can be abused in AI/ML environments. The project also demonstrates how malicious payloads may be concealed inside model tensors using LSB steganography techniques.

This repository was developed as part of practical adversarial machine learning and AI security research focused on model supply chain risks and unsafe serialization mechanisms.

---

## Objectives

The purpose of this project is to demonstrate:

- Unsafe deserialization vulnerabilities in PyTorch
- Arbitrary code execution through malicious model files
- Abuse of Python pickle serialization
- Payload embedding within model tensors
- AI/ML supply chain attack concepts
- Security risks of untrusted model ingestion pipelines

---

## Technical Summary

The attack chain implemented in this project follows these stages:

1. Build and train a simple PyTorch neural network
2. Save a legitimate `state_dict`
3. Generate a malicious payload
4. Embed payload data into model tensors using LSB steganography
5. Construct a malicious wrapper object leveraging pickle deserialization
6. Save the trojanized `.pth` artifact
7. Upload the model to a vulnerable service
8. Trigger code execution during deserialization

---

## Technologies Used

- Python 3
- PyTorch
- Python Pickle Serialization
- Tensor Steganography
- Flask
- Adversarial Machine Learning Concepts

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── pytorch_model_deserialization_rce.ipynb
├── screenshots/
└── images/
```

---

## Vulnerable Code Example

```python
import torch

model = torch.load("uploaded_model.pth")
```

PyTorch model loading relies on Python pickle deserialization, which can execute arbitrary Python code if untrusted model files are loaded without validation or sandboxing.

---

## Security Impact

Successful exploitation may allow an attacker to:

- Execute arbitrary system commands
- Establish remote access
- Compromise ML infrastructure
- Deploy persistence mechanisms
- Exfiltrate sensitive information
- Poison downstream AI pipelines
- Abuse trusted model distribution workflows

---

## Defensive Considerations

Recommended mitigations include:

- Never loading untrusted model artifacts
- Avoiding unsafe pickle deserialization when possible
- Using safer serialization formats such as:
  - `safetensors`
  - ONNX
- Sandboxing model execution environments
- Validating uploaded model artifacts
- Restricting outbound network communication
- Implementing signed model verification workflows

---

## Educational Purpose

This repository was created strictly for:

- Cybersecurity education
- AI security research
- Adversarial ML training
- Demonstrating ML supply chain risks

This project is intended for authorized security research and educational use only.

---
## Demonstration

### Successful Reverse Shell Execution

<img width="2114" height="605" alt="image" src="https://github.com/user-attachments/assets/a2163578-7a45-4fd7-8ea2-bd664ba95d02" />


## References

- PyTorch Documentation
- OWASP Machine Learning Security Top 10
- Python Pickle Serialization Documentation
- Adversarial Machine Learning Research

---

DJD
