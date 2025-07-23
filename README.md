# 📧🛡️🧠 Spam Email Filter Model using NLP

This project demonstrates an end-to-end **Spam Email Classification** system using **Natural Language Processing (NLP)** techniques and **Naive Bayes** classifier. The goal is to predict whether an email is spam (`1`) or not spam (`0`) based on the **subject line** of the email.

---

## 📑 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🗃️ Dataset Overview](#-dataset-overview)
- [⚙️ Setup](#-setup)
- [📚🗂️ Import Libraries and Dataset](#-import-libraries-and-dataset)
- [🔢🧠 Count Vectorization](#-count-vectorization)
- [🧠⚙️ Model Training](#-model-training)
- [📊📉 Model Evaluation](#-model-evaluation)
- [✅ Results](#-conclusions)
- [🤝 Contributing](#-contributing)

---

## 🚀 Project Overview

This notebook builds a **spam email filter** using machine learning and basic NLP techniques. It walks through:

- Cleaning and preparing textual email subject data.
- Vectorizing text using **CountVectorizer**.
- Training a model using **Multinomial Naive Bayes**.
- Evaluating the classifier using **confusion matrix** and predictions.

---

## 🗃️ Dataset Overview

- 📝 The dataset contains two columns:
  - `Subject`: Subject line of the email
  - `Spam`: Binary label (1 = spam, 0 = not spam)
- 📌 Small, structured, and ideal for learning binary classification with textual data.

---

## ⚙️ Setup

### 🔧 Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### 🧰 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
### 📂 Clone this repo:

```bash
https://github.com/Mehardeep79/Traffic-Signals-LENET-Classification-Model.git
cd Traffic-Signals-LENET-Classification-Model
```
### 📂 Download the Dataset:

📍 **Source:** [Kaggle — Spam email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)  

## 📚🗂️ Import Libraries and Dataset

We begin by importing essential libraries and loading the spam email dataset.

---

## 🔢🧠 Count Vectorization

The text data (email subject) is transformed into a numerical format using **CountVectorizer**, which counts the frequency of each word (token) in the dataset to create feature vectors.

---

## 🧠⚙️ Model Training

The model is trained using **Multinomial Naive Bayes**, in two different approaches:

- **Strategy 1**: Train on the full dataset and test on manually input custom data.
- **Strategy 2**: Split the dataset into training and testing sets to simulate real-world performance.

---

## 📊📉 Model Evaluation

The model’s predictions are evaluated using a **confusion matrix**:

- 📘 Light-colored boxes show correct predictions.
- 📕 Dark-colored boxes show misclassifications.

Training and testing performance are both analyzed.

---

## ✅ Results

- 🧠 The Naive Bayes model performs well on structured spam classification.
- 💡 99% accuracy achieved on test samples.
- 🚀 Potential improvements:
  - Add **TF-IDF** vectorization
  - Try **Logistic Regression** or **SVM**
  - Expand the dataset with real-world email subjects

---

## 🤝 Contributing

Feel free to fork this repository and suggest improvements through pull requests. You can help enhance:

- Data cleaning
- Feature engineering
- Model performance
- UI and deployment

