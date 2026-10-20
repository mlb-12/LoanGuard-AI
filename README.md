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

## Live Application

**Live demo:** [Home Credit Default Risk Assessment](https://loan-guard---ai-ekl4mpd5rq6vjvmgtkhvxm.streamlit.app/)

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

```text
LoanGuard-AI/
├── app/                   # Streamlit application
├── data/                  # Home Credit CSV datasets (download separately)
├── explainability/        # Reference data for LIME explanations
├── models/                # Saved model and preprocessing artifacts
├── notebooks/             # EDA, preprocessing, modeling, and explainability notebooks
├── reports/               # Data quality and project reports
├── src/                   # Prediction, preprocessing, feature engineering, and training code
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

LoanGuard AI uses data from Kaggle's **Home Credit Default Risk** competition.

**Dataset page:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

You need a Kaggle account and may need to accept the competition rules before downloading the files. The dataset is not bundled with this repository.

### Dataset files

Download and extract the competition data. The project data folder should contain the CSV files used by the notebooks and application, including:

| File | Description |
|---|---|
| `application_train.csv` | Applicant data with the target label, used for training and analysis. |
| `application_test.csv` | Applicant data without the target label. |
| `bureau.csv` | Previous credits reported by other financial institutions. |
| `bureau_balance.csv` | Monthly status history for bureau credit records. |
| `previous_application.csv` | Previous applications submitted to Home Credit. |
| `POS_CASH_balance.csv` | Historical point-of-sale and cash-loan records. |
| `installments_payments.csv` | Installment payment history. |
| `credit_card_balance.csv` | Historical credit-card balance records. |
| `HomeCredit_columns_description.csv` | Descriptions of dataset columns. |
| `sample_submission.csv` | Example submission format from the competition. |

### Download and place the files

1. Open the [Kaggle dataset page](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Sign in to Kaggle and accept the competition rules if prompted.
3. Download the competition data ZIP file.
4. Extract the ZIP file.
5. Copy the required CSV files directly into the repository's `data/` directory.

The expected paths should look like this:

```text
LoanGuard-AI/
└── data/
    ├── application_train.csv
    ├── application_test.csv
    ├── bureau.csv
    ├── bureau_balance.csv
    ├── credit_card_balance.csv
    ├── HomeCredit_columns_description.csv
    ├── installments_payments.csv
    ├── POS_CASH_balance.csv
    ├── previous_application.csv
    └── sample_submission.csv
```

Place the CSVs directly inside `data/`, not inside an extra nested folder. For example, the expected path is `data/application_train.csv`.

## Installation

### Prerequisites

- Python
- Git
- Visual Studio Code or another Python-compatible editor
- Jupyter support if you plan to run the notebooks

### 1. Clone the repository

```bash
git clone <https://github.com/mlb-12/LoanGuard-AIL>
cd LoanGuard-AI
```
### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Complete the steps in the **Dataset** section and verify that the required files are present in `data/`.

## Preprocessing and Notebooks

The repository includes notebooks for exploring the data, preparing features, training models, and explaining predictions.

Recommended workflow:

1. **Exploratory Data Analysis (EDA):** inspect table dimensions, data types, missing values, duplicate rows, target distribution, and unusual values.
2. **Preprocessing and feature engineering:** apply the transformations required by the training pipeline, including missing-value handling and categorical/numerical feature processing.
3. **Modeling and tuning:** train and evaluate models using the prepared features.
4. **Explainability:** inspect model explanations and feature contributions.

The notebooks are located in `notebooks/`. Open them in Visual Studio Code or Jupyter and run their cells in the intended order. Check the notebook filenames in the repository if they differ from the names shown in your local checkout.

### Important preprocessing notes

- Keep the original downloaded CSV files unchanged where possible; create processed outputs separately.
- Do not assume raw CSV files can be passed directly to the trained model.
- Apply the same preprocessing transformations during training and prediction.
- Ensure that feature names and feature order match the saved model configuration.
- Project analysis documents an expected model input of **226 features**. Verify the saved feature configuration and processed model input before making predictions.
- Review missing values and unusual values in context. An extreme value is not automatically an error.

## Run Locally

Run the following command from the project root after installing dependencies and preparing any required model artifacts and data:

```bash
python -m streamlit run app/app.py
```

Streamlit will print a local address in the terminal, typically `http://localhost:8501`. Open that address in your browser to use the local application.

If startup fails, check that the virtual environment is active, dependencies are installed, and the required model and preprocessing artifacts are available at the paths expected by the application.

## Data Quality

The project's data quality report is available at `reports/data_quality_report.md`.

The current EDA and data-quality checks report zero exact duplicate rows across the eight main CSV datasets. They also identify missing values in multiple features, a special `DAYS_EMPLOYED` value, income outliers under the IQR rule, and an imbalanced target class. Refer to the report for the detailed results and remaining validation items.

The model documentation specifies an expected input of 226 features; the saved feature configuration and final processed model input should be verified together before deployment.

## Dataset and Repository Notes

The Home Credit CSV files are large and should be downloaded separately from Kaggle rather than committed to Git. Check `.gitignore` to ensure raw data, local virtual environments, secrets, and notebook checkpoint files are excluded as appropriate.

Do not commit API keys, passwords, `.env` files, or other credentials to the repository.

## Disclaimer

This project is for educational and research purposes. Predictions are estimates from a machine-learning model and should not be used as the sole basis for loan approval, rejection, or other consequential financial decisions.
