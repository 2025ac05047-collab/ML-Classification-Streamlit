import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer ML Model Evaluation",
    page_icon="🦠",
    layout="wide"
)

with st.sidebar:
    st.title("🦠 Project Controls")

    st.subheader("Dataset")
    st.write("Upload your test dataset and select a model for evaluation.")

    st.divider()

    st.subheader("About")
    st.write(
        "This application compares five classification models "
        "using six evaluation metrics."
    )

    st.divider()

    st.caption("Breast Cancer ML Model Assignment")

st.title("🦠 Breast Cancer ML Model Evaluation Dashboard")
st.write(
    "Interactive evaluation of five machine learning "
    "classification models using test data."
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),

        "KNN": joblib.load(
            MODEL_DIR / "knn.pkl"
        ),

        "Naive Bayes": joblib.load(
            MODEL_DIR / "naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.pkl"
        )
    }

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    return models, scaler


# ============================================================
# LOAD SAVED MODELS
# ============================================================

try:
    models, scaler = load_models()

except Exception as e:
    st.error(
        "Unable to load the saved models. "
        "Please make sure the model folder contains "
        "all required .pkl files."
    )
    st.exception(e)
    st.stop()


# ============================================================
# CSV UPLOAD
# ============================================================

with st.sidebar:
    st.subheader("Upload Test Data")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file to begin."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read the uploaded CSV file.")
    st.exception(e)
    st.stop()


st.success("Test data uploaded successfully!")

st.subheader("Test Data Preview")

st.dataframe(
    test_data.head(),
    use_container_width=True
)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if "Diagnosis" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a 'Diagnosis' column."
    )

    st.stop()


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test_app = test_data.drop(
    columns=["Diagnosis"]
)

y_test_app = test_data["Diagnosis"].map({
    "B": 0,
    "M": 1
})


# Check target values

if y_test_app.isnull().any():

    st.error(
        "The Diagnosis column must contain only "
        "'B' (Benign) or 'M' (Malignant)."
    )

    st.stop()


# ============================================================
# MODEL SELECTION
# ============================================================

st.header("2. Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# FUNCTION FOR MODEL INPUT
# ============================================================

def prepare_input(model_name, X):

    if model_name in [
        "Logistic Regression",
        "KNN"
    ]:

        return scaler.transform(X)

    return X


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name):

    model = models[model_name]

    X_input = prepare_input(
        model_name,
        X_test_app
    )

    predictions = model.predict(X_input)

    probabilities = model.predict_proba(
        X_input
    )[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test_app,
            predictions
        ),

        "AUC": roc_auc_score(
            y_test_app,
            probabilities
        ),

        "Precision": precision_score(
            y_test_app,
            predictions
        ),

        "Recall": recall_score(
            y_test_app,
            predictions
        ),

        "F1 Score": f1_score(
            y_test_app,
            predictions
        ),

        "MCC": matthews_corrcoef(
            y_test_app,
            predictions
        )
    }

    return predictions, probabilities, metrics


# ============================================================
# SELECTED MODEL RESULTS
# ============================================================

y_pred, y_prob, selected_metrics = evaluate_model(
    selected_model_name
)


st.header("3. Evaluation Metrics")

st.subheader(
    f"Results: {selected_model_name}"
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{selected_metrics['Accuracy']:.4f}"
)

col2.metric(
    "AUC Score",
    f"{selected_metrics['AUC']:.4f}"
)

col3.metric(
    "Precision",
    f"{selected_metrics['Precision']:.4f}"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Recall",
    f"{selected_metrics['Recall']:.4f}"
)

col5.metric(
    "F1 Score",
    f"{selected_metrics['F1 Score']:.4f}"
)

col6.metric(
    "MCC Score",
    f"{selected_metrics['MCC']:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y_test_app,
    y_pred
)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(
    f"{selected_model_name} - Confusion Matrix"
)

st.pyplot(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("5. Classification Report")

report = classification_report(
    y_test_app,
    y_pred,
    target_names=[
        "Benign",
        "Malignant"
    ],
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("6. Model Comparison")

comparison_results = {}

for model_name in models.keys():

    _, _, metrics = evaluate_model(
        model_name
    )

    comparison_results[model_name] = metrics


comparison_df = pd.DataFrame(
    comparison_results
).T

comparison_df = comparison_df[
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
]

comparison_df = comparison_df.round(4)

st.dataframe(
    comparison_df,
    use_container_width=True
)


# ============================================================
# OVERALL WINNER
# ============================================================

winner = comparison_df[
    "F1 Score"
].idxmax()

st.subheader("🏆 Overall Winner")

st.success(
    f"{winner} has the highest F1 Score "
    f"among the five implemented models."
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.header("7. Prediction Results")

prediction_results = test_data.copy()

prediction_results["Predicted Diagnosis"] = [
    "M" if value == 1 else "B"
    for value in y_pred
]

prediction_results["Prediction Correct"] = (
    prediction_results["Diagnosis"]
    ==
    prediction_results["Predicted Diagnosis"]
)

st.dataframe(
    prediction_results,
    use_container_width=True
)
st.subheader("📊 Compare Model Performance")

metric_to_plot = st.selectbox(
    "Select a metric to compare:",
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

chart_data = comparison_df[[metric_to_plot]]

st.bar_chart(chart_data)