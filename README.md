# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for identifying different varieties of dry beans based on their physical characteristics.

The dataset contains measurements describing the shape and dimensions of dry bean samples. The target variable is `Class`, which represents the bean variety.

This project also includes an interactive Streamlit application that allows users to upload test data and evaluate the trained classification models.

---

## 2. Dataset Description

The Dry Bean dataset contains 13,611 observations and 17 columns.

- 16 input features
- 1 target variable: `Class`
- No missing values were present in the dataset.
- 68 duplicate rows were identified and removed during preprocessing.

### Features

The dataset contains the following features:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

### Target Classes

The target variable contains seven dry bean classes:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded and inspected the dataset.
2. Checked the dataset dimensions and data types.
3. Checked for missing values.
4. Identified and removed duplicate records.
5. Separated the features (`X`) and target (`y`).
6. Split the dataset into training and testing sets using stratified sampling.
7. Applied feature scaling where required.

After preprocessing:

- Training samples: 10,834
- Testing samples: 2,709
- Number of features: 16

---

## 4. Classification Models

Five machine learning classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

The models were evaluated using the same test dataset for a fair comparison.

---

## 5. Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Classification Report
- Confusion Matrix

---

## 6. Model Comparison

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| k-Nearest Neighbors | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Random Forest | 0.9199 | 0.9907 | 0.9199 | 0.9199 | 0.9198 | 0.9031 |

---

## 7. Observations

Random Forest achieved the highest overall classification performance with an accuracy of 91.99%, F1 score of 91.98%, and MCC score of 0.9031.

Logistic Regression performed very similarly and achieved the highest AUC score of 0.9934.

k-Nearest Neighbors also performed well, achieving an accuracy of 91.55%.

Decision Tree achieved an accuracy of 89.66%, which was lower than Random Forest, Logistic Regression, and kNN.

Gaussian Naive Bayes produced the lowest overall performance, with an accuracy of 76.30%.

The BOMBAY class was classified particularly well by all five models, achieving perfect precision and recall in the test results.

Overall, Random Forest provided the best balance across the evaluation metrics and was the best-performing model in this experiment.

---

## 8. Streamlit Application

An interactive Streamlit application was developed to evaluate the trained models.

The application allows the user to:

- Upload the test dataset in CSV format.
- Preview the uploaded dataset.
- Select one of the five classification models.
- Generate predictions using the selected model.
- View classification performance and evaluation results.

### Live Application

https://dry-bean-classification-app-qljsxpboe63ygxj6izhaqr.streamlit.app/

---

## 9. Repository Structure

```text
dry-bean-classification-streamlit/
│
├── MLAssignment2.ipynb
├── app.py
├── requirements.txt
├── test_data.csv
├── README.md
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
