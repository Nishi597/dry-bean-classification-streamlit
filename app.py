import streamlit as st
import pandas as pd
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
    page_title="Dry Bean Model Lab",
    page_icon="🫘",
    layout="wide"
)

# --------------------------------------------------
# PROJECT-SPECIFIC CONTENT
# --------------------------------------------------

EXPECTED_FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

MODEL_INFO = {
    "Logistic Regression": {
        "summary": "A linear probabilistic classifier used here as a strong multiclass baseline.",
        "preprocessing": "Uses standardized features."
    },
    "Decision Tree": {
        "summary": "A rule-based classifier that splits the feature space into decision regions.",
        "preprocessing": "Uses the original feature scale."
    },
    "k-Nearest Neighbors (kNN)": {
        "summary": "A distance-based classifier that predicts from nearby training examples.",
        "preprocessing": "Uses standardized features because distance is scale-sensitive."
    },
    "Gaussian Naive Bayes": {
        "summary": "A probabilistic classifier that models each numeric feature with a Gaussian distribution.",
        "preprocessing": "Uses the original numeric features."
    },
    "Random Forest": {
        "summary": "An ensemble of decision trees that combines many tree predictions.",
        "preprocessing": "Uses the original feature scale."
    }
}

BENCHMARK_RESULTS = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "k-Nearest Neighbors (kNN)",
        "Gaussian Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [0.9192, 0.8966, 0.9155, 0.7630, 0.9199],
    "AUC": [0.9934, 0.9363, 0.9811, 0.9644, 0.9907],
    "Precision": [0.9197, 0.8965, 0.9163, 0.7647, 0.9199],
    "Recall": [0.9192, 0.8966, 0.9155, 0.7630, 0.9199],
    "F1": [0.9193, 0.8964, 0.9157, 0.7607, 0.9198],
    "MCC": [0.9023, 0.8750, 0.8978, 0.7143, 0.9031]
})

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "k-Nearest Neighbors (kNN)": joblib.load("model/knn.pkl"),
        "Gaussian Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")
    return models, scaler


models, scaler = load_models()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🫘 Dry Bean Model Lab")
st.caption(
    "Interactive evaluation of five machine-learning classifiers trained on "
    "the Dry Bean dataset."
)

intro_left, intro_mid, intro_right = st.columns(3)

with intro_left:
    st.metric("Bean classes", "7")

with intro_mid:
    st.metric("Input features", "16")

with intro_right:
    st.metric("Training models", "5")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Experiment Controls")

selected_model_name = st.sidebar.selectbox(
    "Select a trained classifier",
    list(models.keys())
)

st.sidebar.markdown(
    f"**How this model works**  \n"
    f"{MODEL_INFO[selected_model_name]['summary']}"
)

st.sidebar.info(
    MODEL_INFO[selected_model_name]["preprocessing"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload the labelled test CSV",
    type=["csv"],
    help="The CSV should contain the 16 feature columns and the target column named 'Class'."
)

st.sidebar.divider()
st.sidebar.caption(
    "Dry Bean classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA."
)

# --------------------------------------------------
# BENCHMARK OVERVIEW
# --------------------------------------------------

with st.expander("View benchmark comparison from the original test split"):
    st.dataframe(
        BENCHMARK_RESULTS.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        "Random Forest achieved the best overall balance of metrics, while "
        "Logistic Regression achieved the highest AUC."
    )

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is None:
    st.subheader("Evaluate a model")
    st.write(
        "Upload `test_data.csv` from the sidebar. The application will validate "
        "the file, run the selected classifier, and display performance and error analysis."
    )
    st.info("Waiting for a labelled Dry Bean test CSV.")
    st.stop()

test_df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

required_columns = EXPECTED_FEATURES + ["Class"]
missing_columns = [col for col in required_columns if col not in test_df.columns]
unexpected_columns = [col for col in test_df.columns if col not in required_columns]

if missing_columns:
    st.error(
        "The uploaded file is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if unexpected_columns:
    st.warning(
        "Extra columns were detected and will be ignored: "
        + ", ".join(unexpected_columns)
    )

test_df = test_df[required_columns]

X_uploaded = test_df[EXPECTED_FEATURES]
y_uploaded = test_df["Class"]

selected_model = models[selected_model_name]

# --------------------------------------------------
# MODEL-SPECIFIC PREPROCESSING
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

accuracy = accuracy_score(y_uploaded, y_pred)

try:
    auc = roc_auc_score(
        y_uploaded,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=selected_model.classes_
    )
except ValueError:
    auc = None

precision = precision_score(
    y_uploaded,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_uploaded,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_uploaded,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(y_uploaded, y_pred)

prediction_df = X_uploaded.copy()
prediction_df["Actual Class"] = y_uploaded.values
prediction_df["Predicted Class"] = y_pred
prediction_df["Correct Prediction"] = (
    prediction_df["Actual Class"] == prediction_df["Predicted Class"]
)

# --------------------------------------------------
# TABBED RESULTS
# --------------------------------------------------

overview_tab, performance_tab, error_tab = st.tabs(
    ["Dataset & Predictions", "Model Performance", "Error Analysis"]
)

with overview_tab:
    st.subheader("Uploaded test dataset")

    data_col1, data_col2, data_col3 = st.columns(3)
    data_col1.metric("Rows", f"{test_df.shape[0]:,}")
    data_col2.metric("Columns", test_df.shape[1])
    data_col3.metric("Target classes present", y_uploaded.nunique())

    st.dataframe(
        test_df.head(10),
        use_container_width=True
    )

    st.subheader("Prediction preview")
    st.dataframe(
        prediction_df[
            ["Actual Class", "Predicted Class", "Correct Prediction"]
        ].head(20),
        use_container_width=True
    )

    prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction results",
        data=prediction_csv,
        file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv"
    )

with performance_tab:
    st.subheader(f"{selected_model_name} performance")

    metric_row1 = st.columns(3)
    metric_row1[0].metric("Accuracy", f"{accuracy:.4f}")
    metric_row1[1].metric(
        "AUC Score",
        f"{auc:.4f}" if auc is not None else "N/A"
    )
    metric_row1[2].metric("Precision", f"{precision:.4f}")

    metric_row2 = st.columns(3)
    metric_row2[0].metric("Recall", f"{recall:.4f}")
    metric_row2[1].metric("F1 Score", f"{f1:.4f}")
    metric_row2[2].metric("MCC Score", f"{mcc:.4f}")

    st.caption(
        "Precision, recall and F1 are weighted averages because the dataset "
        "contains seven classes with different class frequencies."
    )

    st.subheader("Classification report")
    report = classification_report(
        y_uploaded,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

with error_tab:
    st.subheader("Confusion matrix")

    class_names = list(selected_model.classes_)

    cm = confusion_matrix(
        y_uploaded,
        y_pred,
        labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix - {selected_model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

    # Most common off-diagonal error
    cm_error = cm.copy()
    for i in range(len(class_names)):
        cm_error[i, i] = 0

    max_error = cm_error.max()

    if max_error > 0:
        error_row, error_col = divmod(cm_error.argmax(), cm_error.shape[1])
        st.info(
            f"Most frequent error for this upload: "
            f"{class_names[error_row]} was predicted as "
            f"{class_names[error_col]} {max_error} times."
        )
    else:
        st.success("No misclassifications were found in the uploaded data.")
