# 🎮 Steam Reviews Sentiment Analysis

### End-to-End NLP & Machine Learning Pipeline

## 🚀 Overview

This project implements a complete end-to-end sentiment analysis pipeline using real user reviews from *Call of Duty: Black Ops 6*, extracted directly from the Steam API.

The objective was to design, evaluate, and compare multiple machine learning models for multi-class sentiment classification (positive, neutral, negative), including handling class imbalance through both oversampling and undersampling techniques.

This project demonstrates skills in:

* Data extraction from APIs
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering with TF-IDF
* Class imbalance handling
* Model evaluation and comparison
* NLP-based classification workflows

---

# 🎯 Problem Statement

User reviews contain valuable insights but are unstructured text data.

The goal of this project was to:

* Extract real user reviews from Steam
* Clean and preprocess textual data
* Automatically classify sentiment
* Compare different ML algorithms
* Evaluate model robustness under class imbalance conditions

---

# 🛠 Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-Learn**
* **TextBlob**
* **Imbalanced-Learn**
* **Matplotlib / Seaborn**
* **WordCloud**
* **Steam Web API**

---

# 🧠 Project Architecture

## 1️⃣ Data Extraction

* Reviews were collected using the Steam Reviews API.
* Only English-language reviews were selected.
* Relevant fields extracted:

  * `review`
  * `voted_up`

Data was exported to CSV for further processing.

---

## 2️⃣ Data Cleaning & Preprocessing

Steps performed:

* Removal of URLs and special characters
* Noise filtering
* Duplicate analysis (intentionally preserved for sentiment relevance)
* Text normalization
* Tokenization via TF-IDF vectorization

A word cloud was generated to identify dominant lexical patterns.

---

## 3️⃣ Sentiment Labeling

Initial sentiment polarity was computed using **TextBlob**:

* Polarity < 0 → Negative
* Polarity = 0 → Neutral
* Polarity > 0 → Positive

This generated a labeled dataset for supervised learning.

---

## 4️⃣ Handling Class Imbalance

To ensure robust model performance:

* 🔼 Random Oversampling
* 🔽 Random Undersampling

Both techniques were evaluated separately to compare their impact on performance.

---

## 5️⃣ Feature Engineering

* TF-IDF Vectorization
* Train/Test split (80/20)
* Cross-validation (5-fold)

---

# 🤖 Models Evaluated

The following classifiers were implemented and compared:

* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression
* Naive Bayes (BernoulliNB)

Each model was evaluated under:

* Oversampling
* Undersampling

Metrics used:

* Accuracy
* Precision (macro)
* Recall
* F1-score
* Confusion Matrix

---

# 📊 Results

Key findings:

* **SVM and Logistic Regression achieved the best performance**
* Oversampling generally improved model stability
* The dataset showed relatively balanced sentiment distribution
* Linear models performed strongly in this TF-IDF feature space

Best performance observed:

* ~0.878 accuracy (oversampling)
* ~0.814 accuracy (undersampling)

This suggests that user sentiment around the game is consistent and well-defined in textual form.

---

# 📈 What This Project Demonstrates

✔ Real-world API data ingestion
✔ NLP preprocessing pipeline design
✔ Feature engineering using TF-IDF
✔ Handling imbalanced datasets
✔ Cross-validation and model comparison
✔ Performance evaluation and interpretation
✔ End-to-end ML workflow

---

# 🚀 How to Run

```bash
pip install pandas numpy scikit-learn textblob imbalanced-learn matplotlib seaborn wordcloud requests
```

Then run the Jupyter Notebook:

```bash
jupyter notebook AlexisVargas_TrabajoFinal_CCD2025.ipynb
```

---

# 🔍 Key Learnings

* Linear models perform strongly in high-dimensional sparse spaces (TF-IDF).
* Proper handling of class imbalance significantly affects model performance.
* API-based data pipelines allow scalable real-world experimentation.
* Model comparison is crucial for selecting production-ready approaches.

---

# 📌 Future Improvements

* Replace TextBlob labeling with manually validated labels
* Implement deep learning models (LSTM / Transformers)
* Deploy as a REST API
* Build interactive dashboard (Streamlit or Flask)
* Perform hyperparameter tuning with GridSearchCV
