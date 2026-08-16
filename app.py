
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.title("🫘 Dry Bean Classification")
st.write(
    "Upload the test dataset and select a machine learning model "
    "to evaluate its classification performance."
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression":
            joblib.load("model/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("model/decision_tree.pkl"),

        "k-Nearest Neighbors (kNN)":
            joblib.load("model/knn.pkl"),

        "Gaussian Naive Bayes":
            joblib.load("model/naive_bayes.pkl"),

        "Random Forest":
            joblib.load("model/random_forest.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is not None:

    # Load uploaded dataset
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.write("Dataset shape:", test_df.shape)

    st.dataframe(
        test_df.head(10),
        use_container_width=True
    )

    # Check target column
    if "Class" not in test_df.columns:

        st.error(
            "The uploaded dataset must contain a 'Class' column."
        )

    else:

        # Separate features and target
        X_uploaded = test_df.drop("Class", axis=1)
        y_uploaded = test_df["Class"]

        selected_model = models[selected_model_name]

        # --------------------------------------------------
        # APPLY SCALING WHERE REQUIRED
        # --------------------------------------------------

        if selected_model_name in [
            "Logistic Regression",
            "k-Nearest Neighbors (kNN)"
        ]:

            X_input = scaler.transform(X_uploaded)

        else:

            X_input = X_uploaded

        # --------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------

        y_pred = selected_model.predict(X_input)
        y_prob = selected_model.predict_proba(X_input)

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_uploaded,
            y_pred
        )

        auc = roc_auc_score(
            y_uploaded,
            y_prob,
            multi_class="ovr",
            average="weighted",
            labels=selected_model.classes_
        )

        precision = precision_score(
            y_uploaded,
            y_pred,
            average="weighted"
        )

        recall = recall_score(
            y_uploaded,
            y_pred,
            average="weighted"
        )

        f1 = f1_score(
            y_uploaded,
            y_pred,
            average="weighted"
        )

        mcc = matthews_corrcoef(
            y_uploaded,
            y_pred
        )

        # --------------------------------------------------
        # DISPLAY METRICS
        # --------------------------------------------------

        st.subheader(
            f"Performance - {selected_model_name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        class_names = list(selected_model.classes_)

        cm = confusion_matrix(
            y_uploaded,
            y_pred,
            labels=class_names
        )

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_uploaded,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

else:

    st.info(
        "Upload test_data.csv from the sidebar to begin model evaluation."
    )
