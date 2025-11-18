#  SMS Spam Classifier (Machine Learning Project)

This repository contains a complete **SMS Spam Detection** machine-learning project built using Python, Scikit-Learn, and Naive Bayes.  
It classifies SMS messages as:

- ✔️ **HAM** — normal messages  
- ❌ **SPAM** — unwanted, promotional, or malicious messages  

The model is trained, optimized, evaluated, and saved for reuse.

---

##  Features

- Removes duplicate messages  
- Full custom text preprocessing 
- CountVectorizer with unigrams + bigrams  
- Multinomial Naive Bayes classifier    
- Hyperparameter tuning (GridSearchCV, 5-fold CV)  
- Predicts new messages with probability scores
- Achieved 91% accuracy on the evaluation server

---

##  Project Structure


spam-classifier/
│
├── Spam.ipynb / spam_classifier_v2.ipynb   # Full Jupyter workflow
├── spam_model.joblib                        # Saved trained model
├── sms_spam_collection/                     # Dataset folder
│   └── SMSSpamCollection
├── README.md                                # Project documentation



---

##  How It Works

### **1. Load Dataset & Remove Duplicates**

```python
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

df = df.drop_duplicates()
```

2. Text Preprocessing

Performed manually using NLTK:

 - Lowercase conversion
 - Regex cleaning
 - Word tokenization
 - Stopword removal
 - Stemming
   
```python
def preprocess(msg):
    msg = msg.lower()
    msg = re.sub(r"[^a-z\s$!]", "", msg)
    tokens = word_tokenize(msg)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

df["clean"] = df["message"].apply(preprocess)
```

This generates numerical feature vectors using:

Unigrams
Bigrams

3. Feature Extraction (CountVectorizer)

Using unigrams + bigrams and filtering overly common terms:

```python
CountVectorizer(
    ngram_range=(1, 2),
    max_df=0.9
)

```

4. Labels Conversion

Spam → 1
Ham → 0

```python
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)
X = df["clean"]
```


5. Model Training (Pipeline + GridSearchCV)
   
```python
pipeline = Pipeline([
    ("vectorizer", CountVectorizer(ngram_range=(1,2), max_df=0.9)),
    ("classifier", MultinomialNB())
])

param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.2, 0.5, 1.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X, y)
best_model = grid.best_estimator_

Reloading later:

```python
model = joblib.load("spam_model.joblib")
```

6. Save the Final Model

```python
import joblib
joblib.dump(best_model, "spam_model.joblib")
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





