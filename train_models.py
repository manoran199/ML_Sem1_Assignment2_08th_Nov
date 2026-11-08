import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef)

OUTPUT = Path(__file__).resolve().parent
MODEL_DIR = OUTPUT / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target.rename("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))]),
    "Decision Tree": Pipeline([("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": Pipeline([("model", GaussianNB())]),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=2.0, probability=True, random_state=42))]),
}
filenames = {"Logistic Regression":"logistic_regression.joblib", "Decision Tree":"decision_tree.joblib", "KNN":"knn.joblib", "Naive Bayes":"naive_bayes.joblib", "Support Vector Machine":"svm.joblib"}
results=[]
for name, model in models.items():
    model.fit(X_train, y_train)
    pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    results.append({"ML Model Name":name, "Accuracy":accuracy_score(y_test,pred), "AUC":roc_auc_score(y_test,prob), "Precision":precision_score(y_test,pred,average="weighted",zero_division=0), "Recall":recall_score(y_test,pred,average="weighted",zero_division=0), "F1":f1_score(y_test,pred,average="weighted",zero_division=0), "MCC":matthews_corrcoef(y_test,pred)})
    joblib.dump(model, MODEL_DIR / filenames[name])
comparison=pd.DataFrame(results).sort_values("F1",ascending=False)
comparison.to_csv(OUTPUT / "model_comparison.csv",index=False)
pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)],axis=1).to_csv(OUTPUT / "test_data.csv",index=False)
print(comparison.to_string(index=False))
