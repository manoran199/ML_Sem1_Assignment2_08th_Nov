from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as snsa
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="ML Classification Dashboard", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification Dashboard")
st.caption("Upload labeled test data, select a saved model, and evaluate it without retraining.")
BASE=Path(__file__).resolve().parent
MODEL_FILES={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","KNN":"knn.joblib","Naive Bayes":"naive_bayes.joblib","Support Vector Machine":"svm.joblib"}
model_name=st.selectbox("Select classification model",list(MODEL_FILES))
upload=st.file_uploader("Upload test CSV containing all 30 features and the target column",type=["csv"])
if upload is None:
    st.info("Upload test_data.csv from this repository to begin.")
    st.stop()
try:
    df=pd.read_csv(upload)
    if "target" not in df.columns: st.error("Missing required target column: target"); st.stop()
    X=df.drop(columns=["target"]); y=df["target"]
    model=joblib.load(BASE/"model"/MODEL_FILES[model_name])
    expected=list(model.feature_names_in_) if hasattr(model,"feature_names_in_") else list(X.columns)
    missing=[c for c in expected if c not in X.columns]
    extra=[c for c in X.columns if c not in expected]
    if missing or extra:
        st.error(f"Feature mismatch. Missing: {missing or 'None'}; Unexpected: {extra or 'None'}"); st.stop()
    X=X[expected]
    pred=model.predict(X); proba=model.predict_proba(X)[:,1]
    values={"Accuracy":accuracy_score(y,pred),"AUC":roc_auc_score(y,proba),"Precision":precision_score(y,pred,average="weighted",zero_division=0),"Recall":recall_score(y,pred,average="weighted",zero_division=0),"F1":f1_score(y,pred,average="weighted",zero_division=0),"MCC":matthews_corrcoef(y,pred)}
    cols=st.columns(6)
    for col,(k,v) in zip(cols,values.items()): col.metric(k,f"{v:.4f}")
    left,right=st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        cm=confusion_matrix(y,pred,labels=[0,1]); fig,ax=plt.subplots(); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Malignant","Benign"],yticklabels=["Malignant","Benign"],ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)
    with right:
        st.subheader("Classification Report")
        report=classification_report(y,pred,target_names=["Malignant","Benign"],output_dict=True,zero_division=0); st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"),use_container_width=True)
except Exception as exc:
    st.exception(exc)
