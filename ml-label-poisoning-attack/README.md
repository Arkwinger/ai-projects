# ML Label Poisoning Attack Lab

This repository contains a Jupyter notebook completed during an AI security skills assessment focused on adversarial machine learning and data poisoning attacks.

The exercise demonstrates how label flipping can poison a supervised learning model by intentionally corrupting training labels belonging to a target class.

## Topics Covered

- Data poisoning attacks
- Label flipping
- Adversarial machine learning
- Training data integrity
- Model reliability degradation

## Technologies Used

- Python
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Files

```text
notebooks/ml_label_poisoning_attack.ipynb
requirements.txt
```

## Attack Summary

The attack randomly modified a portion of Class 1 training labels:
- 25% relabeled to Class 0
- 25% relabeled to Class 2

This caused the model to learn incorrect class boundaries and reduced classification integrity.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch JupyterLab:

```bash
jupyter lab
```

Open the notebook and run the cells sequentially.

## Disclaimer

This repository is intended for educational and defensive security research purposes only.

## Credits

- Hack The Box Academy
