# 🚀 DecodeLabs Internship Projects — Govind Sharma

> AI & Python projects built during the DecodeLabs Industrial Training Internship | Batch 2026

---

## 📁 Project Overview

| # | Project | Tech Used | Topic |
|---|---------|-----------|-------|
| 1 | Rule-Based AI Chatbot | Python | NLP / Chatbot |
| 2 | Data Classification Using AI | Python, Scikit-learn, Matplotlib | KNN / Machine Learning |
| 3 | Tech Stack Recommender | Python | TF-IDF / Content-Based Filtering |

---

## 🤖 Project 1 — Rule-Based AI Chatbot

A simple rule-based chatbot built using Python dictionaries and pattern matching.

**How it works:**
- User inputs a message
- Bot matches input against a knowledge base (dictionary)
- Returns a predefined response or a fallback message

**Features:**
- Greeting & farewell handling
- AI & Python knowledge responses
- Clean input sanitization
- Infinite conversation loop with quit option

**Run:**
```bash
python project_1.py
```

---

## 📊 Project 2 — Data Classification Using AI (KNN)

A complete machine learning pipeline to classify Iris flower species using the K-Nearest Neighbors algorithm.

**How it works:**
- Loads the Iris dataset (150 samples, 4 features, 3 classes)
- Scales features using StandardScaler
- Finds optimal K using the Elbow Method
- Trains a KNN classifier and evaluates performance

**Features:**
- Feature scaling with StandardScaler
- Elbow method to find best K value
- Confusion matrix & F1 score evaluation
- 6-panel visualization dashboard
- Prediction on new flower samples

**Results:**
- High accuracy on Iris dataset
- Weighted F1 Score computed per class
- Visual output saved as `iris_knn_results.png`

**Requirements:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**Run:**
```bash
python project_2.py
```

---

## 🧠 Project 3 — Tech Stack Recommender

A content-based recommendation system that suggests career paths based on a user's skills using TF-IDF and Cosine Similarity.

**How it works:**
- Loads job roles and their required skills from a CSV file
- Builds TF-IDF vectors for each job role
- Converts user's entered skills into a vector
- Computes cosine similarity to rank best matching careers

**Features:**
- TF-IDF vectorization from scratch (no ML libraries)
- Cosine similarity for matching
- Top 3 career path recommendations
- Visual progress bar for match score
- Cold-start handling for unknown skills

**Requirements:**
- `raw_skills.csv` file with job roles and skills

**Run:**
```bash
python project_3.py
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Concepts:** Rule-Based AI, KNN Classification, TF-IDF, Cosine Similarity

---

## 👨‍💻 Author

**Govind Sharma**
DecodeLabs Industrial Training Internship | Batch 2026

---

## 📜 License

This project is built for educational purposes as part of the DecodeLabs Internship Program.
