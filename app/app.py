
from pathlib import Path
import sys
import json

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# 1. PROJECT PATHS AND IMPORTS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict import predict_applicant
from src.preprocessing import transform_new_applicants

MODEL_DIR = PROJECT_DIR / "models"
EXPLAINABILITY_DIR = PROJECT_DIR / "explainability"


# =========================================================
# 2. LOAD MODEL AND EXPLAINABILITY RESOURCES
# =========================================================

@st.cache_resource
def load_explainability_resources():

    model = joblib.load(
        MODEL_DIR / "loan_default_model.pkl"
    )

    pipeline = joblib.load(
        MODEL_DIR / "preprocessing_pipeline.joblib"
    )

    with open(
        MODEL_DIR / "model_feature_columns.json",
        "r",
        encoding="utf-8"
    ) as f:
        feature_columns = json.load(f)

    background_path = (
        EXPLAINABILITY_DIR / "lime_background.pkl"
    )

    background = pd.read_pickle(background_path)

    background = background.reindex(
        columns=feature_columns
    ).fillna(0)

    shap_explainer = shap.TreeExplainer(model)

    lime_explainer = LimeTabularExplainer(
        training_data=background.to_numpy(),
        feature_names=feature_columns,
        class_names=[
            "No repayment difficulties",
            "Repayment difficulties"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    return (
        model,
        pipeline,
        feature_columns,
        shap_explainer,
        lime_explainer
    )


# =========================================================
# 3. LOAD RESOURCES FOR BATCH PREDICTION
# =========================================================

@st.cache_resource
def load_batch_resources():

    model = joblib.load(
        MODEL_DIR / "loan_default_model.pkl"
    )

    pipeline = joblib.load(
        MODEL_DIR / "preprocessing_pipeline.joblib"
    )

    with open(
        MODEL_DIR / "model_feature_columns.json",
        "r",
        encoding="utf-8"
    ) as f:
        feature_columns = json.load(f)

    with open(
        MODEL_DIR / "selected_threshold.json",
        "r",
        encoding="utf-8"
    ) as f:
        threshold = float(json.load(f)["threshold"])

    return model, pipeline, feature_columns, threshold


def predict_batch_from_csv(applicants_df):
    """
    Predict risk for every applicant in an uploaded CSV.

    The CSV should contain raw applicant features,
    not already-preprocessed model features.
    """

    if applicants_df.empty:
        raise ValueError(
            "The uploaded CSV does not contain any applicants."
        )

    # Work with a copy so the uploaded data is not modified.
    applicants_df = applicants_df.copy()

    # Remove leading/trailing spaces from column names.
    applicants_df.columns = applicants_df.columns.str.strip()

    # Prevent duplicate column names from causing ambiguity.
    if applicants_df.columns.duplicated().any():
        raise ValueError(
            "The CSV contains duplicate column names. "
            "Please correct them and upload the file again."
        )

    # TARGET contains the known outcome in training data.
    # It must not be used as an input when predicting new applicants.
    if "TARGET" in applicants_df.columns:
        applicants_df = applicants_df.drop(columns=["TARGET"])

    # Preserve applicant IDs for the output.
    if "SK_ID_CURR" in applicants_df.columns:
        applicant_ids = (
            applicants_df["SK_ID_CURR"].reset_index(drop=True)
        )
    else:
        applicant_ids = pd.Series(
            range(1, len(applicants_df) + 1),
            name="Applicant_ID"
        )

    model, pipeline, feature_columns, threshold = (
        load_batch_resources()
    )

    # Transform all rows using the existing fitted preprocessing pipeline.
    processed = transform_new_applicants(
        pipeline,
        applicants_df
    )

    # The model must receive features in the exact training order.
    if list(processed.columns) != feature_columns:
        raise ValueError(
            "The processed CSV features do not match the model's "
            "expected feature columns. Check the preprocessing "
            "pipeline and model_feature_columns.json."
        )

    # Predict probability for every applicant.
    probabilities = model.predict_proba(processed)[:, 1]

    # Use the project's selected threshold.
    predicted_classes = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "Applicant_ID": applicant_ids,
        "Default_Probability_Percent": (
            probabilities * 100
        ).round(2),
        "Predicted_Class": predicted_classes,
        "Risk_Label": [
            "Repayment difficulties predicted"
            if prediction == 1
            else "No repayment difficulties predicted"
            for prediction in predicted_classes
        ],
        "Decision_Threshold_Percent": round(
            threshold * 100, 2
        )
    })

    return results


# =========================================================
# 4. CONFIGURE THE STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Home Credit Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Home Credit Default Risk Assessment")

st.write(
    "Estimate the probability of repayment difficulties "
    "for individual applicants or a CSV containing multiple applicants."
)

st.warning(
    "This is an experimental model prediction, not a final "
    "loan approval or rejection decision."
)


# =========================================================
# 5. BATCH INFERENCE: CSV UPLOAD
# =========================================================

st.divider()
st.header("📂 Batch Risk Prediction")
st.write(
    "Upload a CSV file containing multiple applicants to predict "
    "their repayment-difficulty risk in one operation."
)

st.info(
    "Upload raw applicant data with column names matching the "
    "original dataset. Do not upload a file containing only "
    "the already-encoded model features."
)

uploaded_file = st.file_uploader(
    "Choose an applicant CSV file",
    type=["csv"],
    key="batch_csv_upload"
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        st.write("### Preview of uploaded applicants")
        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        st.caption(
            f"Rows in uploaded file: {len(batch_df)} | "
            f"Columns: {len(batch_df.columns)}"
        )

        if st.button(
            "Predict Risk for All Applicants",
            type="primary",
            key="batch_predict_button"
        ):

            with st.spinner(
                "Processing applicants and generating predictions..."
            ):

                batch_results = predict_batch_from_csv(batch_df)

                st.session_state["batch_results"] = batch_results

        if "batch_results" in st.session_state:

            batch_results = st.session_state["batch_results"]

            st.subheader("Batch Prediction Results")

            total_applicants = len(batch_results)

            high_risk_count = int(
                (
                    batch_results["Predicted_Class"] == 1
                ).sum()
            )

            other_count = total_applicants - high_risk_count

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Applicants processed",
                total_applicants
            )

            col2.metric(
                "Predicted repayment difficulties",
                high_risk_count
            )

            col3.metric(
                "Not flagged by model",
                other_count
            )

            st.dataframe(
                batch_results,
                use_container_width=True
            )

            # Download results as a CSV file.
            csv_data = batch_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results (CSV)",
                data=csv_data,
                file_name="loanguard_batch_predictions.csv",
                mime="text/csv",
                key="download_batch_results"
            )

            st.caption(
                "The model's classification threshold is applied to "
                "each applicant. A predicted class is a model output, "
                "not a guaranteed outcome or a loan decision."
            )

    except Exception as error:
        st.error(
            "The CSV could not be processed. Check that it contains "
            "valid raw applicant data and compatible column names."
        )
        st.exception(error)


# =========================================================
# 6. INDIVIDUAL APPLICANT INPUT FORM
# =========================================================

st.divider()
st.header("👤 Individual Applicant Prediction")

with st.form("applicant_form"):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual income",
            min_value=1.0,
            value=180000.0,
            step=10000.0
        )

        credit = st.number_input(
            "Credit amount",
            min_value=1.0,
            value=450000.0,
            step=10000.0
        )

        annuity = st.number_input(
            "Loan annuity (annual payment amount)",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

    with col2:

        goods_price = st.number_input(
            "Price of goods",
            min_value=0.0,
            value=400000.0,
            step=10000.0
        )

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30
        )

        employment_years = st.number_input(
            "Years employed",
            min_value=0.0,
            max_value=70.0,
            value=5.0,
            step=1.0
        )

        family_members = st.number_input(
            "Number of family members",
            min_value=1,
            max_value=20,
            value=3
        )

    submitted = st.form_submit_button(
        "Predict Repayment Risk",
        type="primary"
    )


# =========================================================
# 7. INDIVIDUAL PREDICTION AND EXPLANATIONS
# =========================================================

if submitted:

    applicant_data = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": -int(age * 365),
        "DAYS_EMPLOYED": -int(employment_years * 365),
        "CNT_FAM_MEMBERS": family_members,
    }

    try:

        result = predict_applicant(applicant_data)

        # Display prediction results first.
        st.subheader("Prediction Results")

        probability = float(result["probability"])
        threshold = float(result["threshold"])

        st.metric(
            "Estimated repayment-difficulty probability",
            f"{probability:.2%}"
        )

        st.progress(
            max(0.0, min(1.0, probability))
        )

        st.write(
            f"Selected classification threshold: {threshold:.0%}"
        )

        if result["predicted_class"] == 1:
            st.error(
                "The model predicts repayment difficulties."
            )
        else:
            st.success(
                "The model does not predict repayment difficulties."
            )

        st.caption(
            "This probability is a model estimate, not a guarantee "
            "of future repayment behavior. Missing information may "
            "be imputed by the preprocessing pipeline."
        )

        # Load SHAP and LIME resources only when explanations are needed.
        (
            model,
            pipeline,
            feature_columns,
            shap_explainer,
            lime_explainer
        ) = load_explainability_resources()

        processed = transform_new_applicants(
            pipeline,
            applicant_data
        )

        processed = processed.reindex(
            columns=feature_columns
        )

        if processed.isna().all(axis=1).iloc[0]:

            st.warning(
                "The applicant has no usable feature values. "
                "The prediction is displayed, but explanations "
                "cannot be generated."
            )

        else:

            # SHAP explanation
            st.divider()
            st.subheader(
                "Why did the model make this prediction?"
            )
            st.markdown(
                "### SHAP: Individual feature contributions"
            )

            shap_values = shap_explainer(processed)

            shap.plots.waterfall(
                shap_values[0],
                max_display=12,
                show=False
            )

            shap_fig = plt.gcf()
            st.pyplot(shap_fig)
            plt.close(shap_fig)

            st.caption(
                "SHAP shows how individual features contribute "
                "to the model output. For this binary XGBoost model, "
                "the waterfall may show raw model-score contributions "
                "rather than probability percentages."
            )

            # LIME explanation
            st.markdown("### LIME: Local explanation")

            lime_result = lime_explainer.explain_instance(
                data_row=processed.iloc[0].to_numpy(),
                predict_fn=model.predict_proba,
                num_features=10,
                labels=(1,)
            )

            lime_fig = lime_result.as_pyplot_figure(
                label=1
            )

            st.pyplot(lime_fig)
            plt.close(lime_fig)

            st.caption(
                "LIME approximates the model locally to show "
                "which features influence the repayment-difficulty "
                "prediction. LIME weights are not probabilities."
            )

    except Exception as error:

        st.error(
            "Prediction or explanation failed. "
            "Check the error details below."
        )

        st.exception(error)
