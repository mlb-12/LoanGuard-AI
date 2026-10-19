
from pathlib import Path
import json
import sys
import joblib
import pandas as pd

# Locate the Member 2 project folder
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Ensure Python can import the existing src package
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import transform_new_applicants


MODEL_DIR = PROJECT_DIR / "models"

# Load the saved artifacts
model = joblib.load(MODEL_DIR / "loan_default_model.pkl")
pipeline = joblib.load(MODEL_DIR / "preprocessing_pipeline.joblib")

with open(
    MODEL_DIR / "model_feature_columns.json",
    "r",
    encoding="utf-8"
) as f:
    FEATURE_COLUMNS = json.load(f)

with open(
    MODEL_DIR / "selected_threshold.json",
    "r",
    encoding="utf-8"
) as f:
    THRESHOLD = float(json.load(f)["threshold"])


def predict_applicant(applicant_data):
    """
    Predict repayment-difficulty risk for one applicant.

    applicant_data: dictionary of raw applicant information.

    Returns the predicted class, probability, and threshold.
    """

    # Convert raw inputs into the 226 model-ready features
    processed = transform_new_applicants(
        pipeline,
        applicant_data
    )

    # Verify that feature names and order match the model
    if list(processed.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Processed features do not match the model feature schema."
        )

    # Calculate probability of class 1
    probability = float(model.predict_proba(processed)[0, 1])

    # Apply the team's selected threshold
    predicted_class = int(probability >= THRESHOLD)

    return {
        "predicted_class": predicted_class,
        "risk_label": (
            "Repayment difficulties predicted"
            if predicted_class == 1
            else "No repayment difficulties predicted"
        ),
        "probability": probability,
        "threshold": THRESHOLD,
    }

def predict_batch(applicants_df):
    """
    Predict repayment-difficulty risk for multiple applicants.

    applicants_df: pandas DataFrame with one row per applicant.
    Returns a DataFrame containing predictions for all applicants.
    """
    if not isinstance(applicants_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if applicants_df.empty:
        raise ValueError("The uploaded CSV contains no applicants.")

    # Preserve applicant IDs before preprocessing removes them.
    if "SK_ID_CURR" in applicants_df.columns:
        applicant_ids = applicants_df["SK_ID_CURR"].reset_index(drop=True)
    else:
        applicant_ids = pd.Series(
            range(1, len(applicants_df) + 1),
            name="Applicant_Row"
        )

    # Convert all applicants into the 226 expected model features.
    processed = transform_new_applicants(
        pipeline,
        applicants_df
    )

    # Ensure the feature names and order match the saved model.
    if list(processed.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Processed features do not match the model feature schema."
        )

    # Predict all applicants in one call.
    probabilities = model.predict_proba(processed)[:, 1]
    predicted_classes = (probabilities >= THRESHOLD).astype(int)

    # Prepare downloadable results.
    results = pd.DataFrame({
        "Applicant_ID": applicant_ids,
        "Default_Probability": probabilities,
        "Predicted_Class": predicted_classes,
        "Risk_Label": [
            "Repayment difficulties predicted"
            if prediction == 1
            else "No repayment difficulties predicted"
            for prediction in predicted_classes
        ],
        "Decision_Threshold": THRESHOLD,
    })

    results["Default_Probability"] = (
        results["Default_Probability"] * 100
    ).round(2)

    return results

