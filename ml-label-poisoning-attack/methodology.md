# Methodology

## Overview

This repository examines label poisoning attacks against supervised machine learning classifiers through adversarial manipulation of training data labels.

The research focuses on how intentionally corrupted class labels can alter model learning behavior, distort decision boundaries, and reduce classification reliability during training.

Experiments were conducted in a controlled environment using targeted label flipping techniques against a supervised classification dataset.

---

# Attack Methodology

The poisoning process involved selectively modifying labels belonging to a target class within the training dataset prior to model training.

A percentage of samples originally belonging to Class 1 were intentionally relabeled as alternative classes:

* 25% relabeled to Class 0
* 25% relabeled to Class 2

The modified dataset was then used to retrain the classifier and evaluate the resulting impact on model behavior and prediction accuracy.

The attack simulated a scenario where an adversary gains the ability to partially manipulate training data labels before the learning process begins.

---

# Training Workflow

The workflow followed these stages:

1. Load and preprocess the classification dataset
2. Split the dataset into training and testing subsets
3. Identify the target class for poisoning
4. Randomly modify a percentage of target class labels
5. Retrain the classifier using the poisoned dataset
6. Evaluate classification performance after poisoning
7. Compare clean and poisoned model behavior

---

# Label Flipping Strategy

The poisoning attack used targeted label flipping to introduce false supervision signals into the training process.

Instead of modifying features directly, the attack manipulated the relationship between features and class labels, causing the classifier to learn incorrect associations during training.

This produced degraded class separation and reduced prediction reliability for the poisoned target class.

---

# Observations

The poisoned model demonstrated:

* reduced classification accuracy
* distorted decision boundaries
* degraded target class reliability
* increased prediction instability
* weakened feature-to-class relationships

The experiments showed that even partial corruption of training labels can significantly impact model learning and classification integrity.

The results also highlighted the importance of training data validation and integrity monitoring within machine learning pipelines.

---

# Research Focus

This research explored:

* adversarial machine learning
* data poisoning attacks
* label flipping techniques
* training data integrity
* model reliability degradation
* supervised learning security risks

---

# Technologies Used

* Python
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Conclusion

The experiments demonstrated how machine learning systems remain highly dependent on the integrity of training data.

Even relatively small amounts of adversarial label corruption were capable of degrading model reliability and altering classifier behavior, emphasizing the importance of secure dataset handling and validation procedures in modern ML pipelines.
