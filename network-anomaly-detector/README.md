#  Network Anomaly Detector (NSL-KDD)

A machine learning–based intrusion detection system using the **NSL-KDD dataset**, designed to classify network traffic into multiple attack categories with high accuracy.

---

##  Project Structure

network-anomaly-detector/
│
├── NSL-KDD-Detector.ipynb # Full pipeline: preprocessing → training → evaluation
├── network_anomaly_detection_model.joblib # Saved Random Forest model
├── KDD+.txt # NSL-KDD dataset file
└── README.md # Documentation



---

##  Classification Categories

All network records are mapped into **five classes**:

| Class | Meaning                |
|-------|-------------------------|
| **0** | Normal                 |
| **1** | DoS (Denial of Service)|
| **2** | Probe                  |
| **3** | Privilege Escalation   |
| **4** | Access Attack          |

---

##  Preprocessing Steps

The notebook performs full data preparation:

- ✔ Loaded and inspected NSL-KDD dataset  
- ✔ Created **binary** and **multi-class** labels  
- ✔ One-hot encoded categorical features  
  - `protocol_type`
  - `service`
- ✔ Selected numerical behavioral features  
- ✔ Combined encoded + numerical features into a training matrix  

---

##  Model Training

A **Random Forest Classifier** is used:

```python
RandomForestClassifier(random_state=1337)
```

Dataset Splitting
The dataset is split into training, validation, and test sets as follows:

Train/Test Split
| Split        | Percentage |
| ------------ | ---------- |
| **Training** | 80%        |
| **Test**     | 20%        |

From the 80% Training Portion

| Set            | Percentage (of training) |
| -------------- | ------------------------ |
| **Training**   | 70%                      |
| **Validation** | 30%                      |


##  Model Evaluation

The Random Forest model demonstrated **excellent performance** on both the validation and test sets.

---

###  Overall Performance Metrics

| Metric | Score |
|--------|--------|
| **Accuracy** | ~0.99 |
| **Precision** | ~0.99 |
| **Recall** | ~0.99 |
| **F1-Score** | ~0.99 |

These results indicate strong predictive performance across all attack categories.

---

###  Evaluation Notes

- Both **validation** and **test** results are consistent  
- Model generalizes well and avoids overfitting  
- Multi-class predictions show high accuracy across *Normal, DoS, Probe, Privilege, Access*  

---

###  Visual Results

Confusion matrices and detailed classification reports are included in the notebook for deeper inspection.

##  Saving & Loading the Model

After training, the Random Forest model is saved using `joblib` for reuse and deployment.

###  Save the model
```python
import joblib
joblib.dump(rf_model_multi, "network_anomaly_detection_model.joblib")
````


























