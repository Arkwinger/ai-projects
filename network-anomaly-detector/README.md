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

##  How to Run This Project

Follow these steps to run the anomaly detection notebook and model.

---

### **1. Clone the repository**

```bash
git clone https://github.com/Arkinger/ai-projects.git
cd ai-projects/network-anomaly-detector
````

### **2. Install dependencies**

Install the required Python packages (a virtual environment is recommended):
````bash
pip install -r requirements.txt
````
### **3. Open the Jupyter Notebook**

````bash
jupyter notebook NSL-KDD-Detector.ipynb
````
Run all cells to:

- Load and preprocess the dataset
- Train the Random Forest model
- Evaluate performance (validation + test)
- Save the trained model

### **4. (Optional) Load the Pre-Trained Model**

The repository includes a fully trained model:
````bash
network_anomaly_detection_model.joblib
````

Load it with:

````python
import joblib
model = joblib.load("network_anomaly_detection_model.joblib")
````

### **5. Make Predictions**

Use the model to classify new (properly preprocessed) network traffic samples:
````python
predictions = model.predict(new_data)
print(predictions)
````

---

##  Project Summary

This project demonstrates a full end-to-end machine learning workflow for intrusion detection using the NSL-KDD dataset.  
It includes data preprocessing, feature encoding, model training, evaluation, and model export for real-world use.

This repository can serve as:

- A learning reference for ML-based intrusion detection  
- A foundation for more advanced cybersecurity analytics projects  
- A starting point for extending the model to deep learning or real-time detection  

Feel free to explore, fork, and build upon this project!


---

##  Future Improvements

Planned enhancements for the next versions:

- Add deep learning models (LSTM / Autoencoders)
- Build a full pipeline with real-time inference
- Improve feature engineering for rare attack types
- Add a web dashboard for live traffic monitoring
- Containerize with Docker for deployment

Contributions and suggestions are welcome!










