import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add repo root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------
# Load dataset and artifacts
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("data/synthetic_pupillometry.csv")

@st.cache_data
def load_artifacts():
    artifacts = {}
    base = "artifacts"

    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        model_dir = os.path.join(base, model_name)
        if os.path.exists(model_dir):
            artifacts[model_name] = {
                "report": pd.read_csv(os.path.join(model_dir, "classification_report.csv"), index_col=0),
                "cm": pd.read_csv(os.path.join(model_dir, "confusion_matrix.csv"), index_col=0),
                "model_path": os.path.join(model_dir, "model.pkl")
            }
    return artifacts

@st.cache_data
def load_label_classes():
    return pd.read_csv("artifacts/label_encoder_classes.csv")["0"].tolist()


# ---------------------------------------------------------
# FDA-style SYSTEM-LEVEL narrative generation
# ---------------------------------------------------------
def generate_narrative_fda(df, artifacts):
    from src.llm_claude import call_claude

    severity_dist = df["severity"].value_counts().to_dict()
    diagnosis_dist = df["diagnosis"].value_counts().to_dict()
    feature_summary = df.describe(include="all").to_dict()

    model_summaries = []
    for model_name, metrics in artifacts.items():
        report = metrics["report"]
        cm = metrics["cm"]

        model_summaries.append(
            f"{model_name}:\n"
            f"- Classification report:\n{report.to_string()}\n"
            f"- Confusion matrix:\n{cm.to_string()}\n"
        )

    model_text = "\n".join(model_summaries)

    prompt = f"""
You are drafting text suitable for inclusion in an FDA submission (e.g., 510(k) or De Novo) for a clinical decision support (CDS) tool based on automated pupillometry.

Generate a concise, neutral, FDA-style narrative summarizing the *entire dataset* and *overall model performance*, not an individual patient.

Dataset characteristics:
- Severity distribution: {severity_dist}
- Diagnosis distribution: {diagnosis_dist}
- Feature summary statistics: {feature_summary}

Model performance:
{model_text}

Generate a structured narrative with the following sections:

1. System Overview
2. Model Inputs and Data Characteristics
3. Model Performance Summary
4. Intended Use and Performance Context
5. Limitations and Appropriate Use

Use formal, regulatory-appropriate language.
Return only the narrative text.
"""

    return call_claude(prompt)


# ---------------------------------------------------------
# Streamlit Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Pupillometry Clinical Dashboard", layout="wide")
st.title("🧠 Pupillometry Clinical Dashboard")

df = load_dataset()

# 🔥 FIX: Convert only object columns to string (preserve numeric columns)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)

artifacts = load_artifacts()
label_classes = load_label_classes()

tabs = st.tabs(["Dataset Overview", "Model Performance", "Patient Explorer", "Narrative Summary"])


# ---------------------------------------------------------
# Tab 1: Dataset Overview
# ---------------------------------------------------------
with tabs[0]:
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severity Distribution")
        st.bar_chart(df["severity"].value_counts())

    with col2:
        st.subheader("Diagnosis Distribution")
        st.bar_chart(df["diagnosis"].value_counts())

    st.subheader("NPI Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["npi"].astype(float), kde=True, bins=40, ax=ax)
    st.pyplot(fig)

    st.subheader("Pupil Size (Left vs Right)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x="pupil_left", y="pupil_right", alpha=0.3, ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------------
# Tab 2: Model Performance
# ---------------------------------------------------------
with tabs[1]:
    st.header("Model Performance")

    if not artifacts:
        st.warning("No model artifacts found. Run `python src/model.py` first.")
    else:
        for model_name, metrics in artifacts.items():
            st.subheader(model_name.replace("_", " ").title())

            st.write("Classification Report")
            report_df = metrics["report"].copy()
            report_df.index = report_df.index.astype(str)
            report_df.columns = report_df.columns.astype(str)
            st.dataframe(report_df)

            st.write("Confusion Matrix")
            cm_df = metrics["cm"].copy()
            cm_df.index = cm_df.index.astype(str)
            cm_df.columns = cm_df.columns.astype(str)
            st.dataframe(cm_df)


# ---------------------------------------------------------
# Tab 3: Patient Explorer
# ---------------------------------------------------------
with tabs[2]:
    st.header("Patient Explorer")

    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())
    patient = df[df["patient_id"] == patient_id].iloc[0]

    st.subheader("Patient Information")
    st.write(patient)

    st.subheader("Model Predictions")

    input_df = patient.drop(labels=["patient_id", "severity"]).to_frame().T

    results = []

    for model_name, metrics in artifacts.items():
        model_path = metrics["model_path"]

        if os.path.exists(model_path):
            model = joblib.load(model_path)

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            results.append({
                "Model": model_name,
                "Prediction": label_classes[pred],
                "P(mild)": proba[label_classes.index("mild")],
                "P(moderate)": proba[label_classes.index("moderate")],
                "P(severe)": proba[label_classes.index("severe")]
            })

    if results:
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("No trained models found.")


# ---------------------------------------------------------
# Tab 4: FDA-Style Narrative Summary (SYSTEM LEVEL)
# ---------------------------------------------------------
with tabs[3]:
    st.header("Narrative Summary (FDA-style)")

    if st.button("Generate FDA-Style System Narrative"):
        if not artifacts:
            st.warning("No trained models available to generate a narrative.")
        else:
            with st.spinner("Generating narrative..."):
                narrative_text = generate_narrative_fda(df, artifacts)
            st.subheader("System-Level FDA Narrative")
            st.write(narrative_text)