#  SMS Spam Classifier (Machine Learning Project)

This repository contains a complete **SMS Spam Detection** machine-learning project built using Python, Scikit-Learn, and Naive Bayes.  
It classifies SMS messages as:

- ✔️ **HAM** — normal messages  
- ❌ **SPAM** — unwanted, promotional, or malicious messages  

The model is trained, optimized, evaluated, and saved for reuse.

---

##  Features

- Full text preprocessing pipeline  
- Lowercasing, punctuation removal, stopword removal, stemming  
- Bag-of-Words feature extraction (unigrams + bigrams)  
- Multinomial Naive Bayes classifier  
- Hyperparameter tuning with GridSearchCV  
- Model saved with joblib  
- Predicts new messages with probability scores  

---

##  Project Structure


spam-classifier/
│
├── Spam.ipynb # Jupyter Notebook (full workflow)
├── spam_model.joblib # Saved trained ML model
├── README.md # Project documentation
└── requirements.txt # (optional) dependencies



---

## 🧠 How It Works

### **1. Preprocessing Steps**

- Convert text to lowercase  
- Remove unwanted punctuation/numbers  
- Tokenize into words  
- Remove stopwords  
- Apply stemming  
- Convert tokens back to a cleaned string  

Example:

```python
message = message.lower()
message = re.sub(r"[^a-z\s$!]", "", message)
tokens = word_tokenize(message)
tokens = [word for word in tokens if word not in stop_words]
tokens = [stemmer.stem(word) for word in tokens]
cleaned_message = " ".join(tokens)
```

2. Feature Extraction (Bag-of-Words)

Using CountVectorizer:

```python
vectorizer = CountVectorizer(
    min_df=1,
    max_df=0.9,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["message"])
```

This generates numerical feature vectors using:

Unigrams
Bigrams

3. Model Training

Pipeline
```python
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])
```

Hyperparameter tuning:

```python
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(df["message"], y)
best_model = grid_search.best_estimator_
```

4. Saving the Model
```python
import joblib
joblib.dump(best_model, "spam_model.joblib")
```
Reloading later:

```python
model = joblib.load("spam_model.joblib")
```

5. Running Predictions

```python
msg = ["Congratulations! You won a FREE prize! Click now!"]
prediction = model.predict(msg)
probabilities = model.predict_proba(msg)
```
Example output:

```python
Prediction: spam
Spam Probability: 1.00
Ham Probability: 0.00
```

##  Dataset

**SMS Spam Collection Dataset**  
5,574 labeled messages (spam or ham)

Source:  
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

##  Technologies Used

- Python  
- Scikit-Learn  
- NLTK  
- Pandas  
- NumPy  
- Joblib  
- Jupyter Notebook  

---

##  Future Improvements

- Use TF-IDF instead of CountVectorizer  
- Test Logistic Regression, SVM, or Random Forest  
- Build a FastAPI/Flask server for predictions  
- Deploy online (Render, AWS, Azure)  
- Add a simple web UI  

---

##  License

This project is open-source.





