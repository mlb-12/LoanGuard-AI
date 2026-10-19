
# LoanGuard AI — Home Credit Default Risk Assessment

## Overview

LoanGuard AI is a machine-learning web application that estimates the probability of repayment difficulties for a loan applicant.

The application uses a trained classification model and provides SHAP and LIME explanations to help users understand the factors influencing an individual prediction.

## Features

- Applicant information input form
- Estimated repayment-difficulty probability
- Classification using a selected decision threshold
- SHAP feature-contribution visualization
- LIME local explanation
- Streamlit web interface

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- LIME
- Streamlit
- Matplotlib

## Project Structure

- `app/` — Streamlit application
- `src/` — prediction, preprocessing, feature engineering, and training code
- `models/` — saved model and preprocessing artifacts
- `explainability/` — reference data for LIME explanations
- `notebooks/` — project notebooks
- `reports/` — project reports

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the application from the project root:

```bash
python -m streamlit run app/app.py
```

## Disclaimer

This project is for educational and research purposes. Predictions are estimates from a machine-learning model and should not be used as the sole basis for loan approval, rejection, or other consequential financial decisions.
