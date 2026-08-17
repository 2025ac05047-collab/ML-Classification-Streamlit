# ML Classification Project

## 1. Problem Statement

The objective of this project is to implement and evaluate multiple machine learning classification models on a public classification dataset. Five classification algorithms are trained and evaluated using multiple performance metrics. An interactive Streamlit application is also developed to allow users to upload test data, select a machine learning model, view predictions, and evaluate model performance.

The implemented models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

The models are evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


## 2. Dataset Description

### Dataset Name

Breast Cancer Wisconsin (Diagnostic)

### Dataset Source

UCI Machine Learning Repository

Official dataset:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Dataset Characteristics

- Number of instances: 569
- Number of input features: 30
- Problem type: Binary Classification
- Feature type: Continuous
- Target variable: Diagnosis
- Missing values: None

The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. The features describe characteristics of cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

The target variable contains two classes:

- B = Benign
- M = Malignant

The dataset satisfies the assignment requirement of at least 500 instances and at least 12 features.


## 3. Data Preprocessing

The following preprocessing steps were performed:

1. The ID column was removed because it is an identifier and does not provide predictive information.
2. The target variable `Diagnosis` was converted into binary numerical values:
   - B = 0
   - M = 1
3. The dataset was divided into training and testing sets using an 80:20 split.
4. Stratified splitting was used to maintain the class distribution between training and testing data.
5. StandardScaler was used for Logistic Regression and KNN because these algorithms are sensitive to feature scale.
6. Decision Tree, Gaussian Naive Bayes, and Random Forest were trained using the original feature values.


## 4. Machine Learning Models

### 4.1 Logistic Regression

Logistic Regression was implemented as a baseline linear classification model. Standardized features were used for training.

### 4.2 Decision Tree Classifier

A Decision Tree Classifier was implemented using the training dataset with a fixed random state for reproducibility.

### 4.3 K-Nearest Neighbor Classifier

A KNN classifier with 5 neighbors was implemented. Standardized features were used because distance-based algorithms are sensitive to differences in feature scale.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was implemented because the dataset contains continuous numerical features.

### 4.5 Random Forest Classifier

Random Forest was implemented as the ensemble learning model using 100 decision trees.


## 5. Model Evaluation

All five models were evaluated using the same test dataset and the following six metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |


## 6. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 96.49% and an AUC score of 0.9960. It also achieved a strong F1 score of 0.9512 and MCC score of 0.9245. The model performed very well overall and demonstrated strong discrimination between the two classes.

### Decision Tree

The Decision Tree achieved an accuracy of 92.98%, which was the lowest accuracy among the five models. Its AUC score was 0.9246 and its MCC score was 0.8492. Although the model performed reasonably well, it was weaker than the other models on the majority of the evaluation metrics.

### K-Nearest Neighbor

KNN achieved an accuracy of 95.61% and an AUC score of 0.9823. It achieved a precision of 0.9744 and an F1 score of 0.9383. The model performed strongly but was slightly below Logistic Regression and Random Forest overall.

### Gaussian Naive Bayes

Naive Bayes achieved an accuracy of 93.86% and an AUC score of 0.9934. It achieved perfect precision of 1.0000, but its recall was 0.8333. This indicates that while its positive predictions were highly precise, it missed a relatively larger number of positive cases compared with the stronger-performing models.

### Random Forest

Random Forest achieved the highest accuracy of 97.37%, along with perfect precision of 1.0000. It achieved an F1 score of 0.9630 and the highest MCC score of 0.9442. The model demonstrated the strongest overall performance across the evaluation metrics.


## 7. Overall Winner

Based on the comparison of the evaluation metrics, **Random Forest** was selected as the overall winner.

Random Forest achieved:

- Accuracy: 0.9737
- AUC: 0.9929
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC: 0.9442

The model achieved the highest accuracy, F1 Score, and MCC among the five implemented models while also maintaining perfect precision.


## 8. Streamlit Application

An interactive Streamlit web application was developed to demonstrate the trained classification models.

The application provides the following functionality:

1. Upload test data in CSV format.
2. Preview the uploaded test data.
3. Select one of the five implemented machine learning models.
4. Generate predictions using the selected model.
5. Display the six required evaluation metrics.
6. Display the confusion matrix.
7. Display the classification report.
8. Display a comparison of all five models.
9. Display prediction results for the uploaded test data.

### Implemented Models

- Logistic Regression
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Random Forest


## 9. Project Structure

```text
ML_Classification_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── ML_Experimentation.ipynb
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```
## 10. GitHub Repository

**GitHub Repository:**  
[ML-Classification-Streamlit](https://github.com/2025ac05047-collab/ML-Classification-Streamlit)


## 11. Streamlit Application

**Live Streamlit Application:**  
[ML Classification Streamlit App](https://ml-classification-app-nxtmqhyeienwjedl5zsk.streamlit.app/)
