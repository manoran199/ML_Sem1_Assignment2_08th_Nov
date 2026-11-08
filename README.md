# Machine Learning Classification Project

## A. Problem Statement
This project compares five supervised classification algorithms on one dataset. Every model uses the same stratified 80/20 split. The saved pipelines are evaluated with Accuracy, AUC, weighted Precision, weighted Recall, weighted F1, and Matthews Correlation Coefficient (MCC).

## B. Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Original repository:** UCI Machine Learning Repository
- **Access used:** bundled `load_breast_cancer` dataset in scikit-learn
- **Problem type:** binary classification
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target:** `target` (`0 = malignant`, `1 = benign`)
- **Missing values:** none in the scikit-learn copy

This dataset satisfies the assignment minimum of 500 instances and 12 features.

## C. GitHub Repository Link
https://github.com/manoran199/ML_Sem1_Assignment2_08th_Nov

## D. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifiera
5. Support Vector Machine

## Model Comparison
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Support Vector Machine | 0.9825 | 0.9967 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| KNN | 0.9737 | 0.9884 | 0.9747 | 0.9737 | 0.9735 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |
| Decision Tree | 0.9123 | 0.9387 | 0.9137 | 0.9123 | 0.9127 | 0.8139 |

## Model Performance Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved accuracy 0.9825, AUC 0.9954, F1 0.9825, and MCC 0.9623. |
| Support Vector Machine | Achieved accuracy 0.9825, AUC 0.9967, F1 0.9825, and MCC 0.9623. |
| KNN | Achieved accuracy 0.9737, AUC 0.9884, F1 0.9735, and MCC 0.9442. |
| Naive Bayes | Achieved accuracy 0.9386, AUC 0.9878, F1 0.9384, and MCC 0.8676. |
| Decision Tree | Achieved accuracy 0.9123, AUC 0.9387, F1 0.9127, and MCC 0.8139. |

## Overall Winner
**Logistic Regression** is the winner under the stated selection rule because it obtained the highest weighted F1 score on the fixed test set. AUC and MCC should also be checked when discussing the result.

## Reproducibilitay
- Test size: 20%
- Split: stratified
- Random state: 42
- Scaling: StandardScaler inside pipelines for Logistic Regression, KNN, and SVM
- Streamlit only loads saved `.joblib` pipelines; it does not retrain on uploaded data.

## Local Execution
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

