# Data Quality Report

## 1. Introduction

This report documents the data quality checks performed for the LoanGuard-AI Home Credit Default Risk prediction project. Data quality matters because missing values, duplicate records, unusual numerical values, and inconsistent features can affect model training, evaluation, and predictions.

The findings below are based on the available Home Credit CSV files and the outputs from the exploratory data analysis (EDA) notebook. Checks that have not been run against the processed model input are explicitly identified as pending.

## 2. Dataset Overview

The project uses data from the Home Credit Default Risk prediction task to predict whether an applicant may experience repayment difficulties.

The EDA notebook reports the following source dataset dimensions:

| Dataset | Rows | Columns |
|---|---:|---:|
| `application_train.csv` | 307,511 | 122 |
| `application_test.csv` | 48,744 | 121 |
| `bureau.csv` | 1,716,428 | 17 |
| `bureau_balance.csv` | 27,299,925 | 3 |
| `previous_application.csv` | 1,670,214 | 37 |
| `POS_CASH_balance.csv` | 10,001,358 | 8 |
| `installments_payments.csv` | 13,605,401 | 8 |
| `credit_card_balance.csv` | 3,840,312 | 23 |

The EDA notebook creates additional engineered columns in its in-memory training DataFrame. Therefore, its later 126-column shape should not be confused with the original `application_train.csv`, which contains 122 columns.

The project documentation indicates that the saved model feature configuration expects 226 input features. The exact feature list and the final processed matrix still need to be verified against the actual model configuration and preprocessing pipeline.

## 3. Data Quality Checks

### 3.1 Missing Values

The EDA notebook reports that the original training dataset has missing values in 67 of its 122 columns. Of these, 41 columns have more than 50% of their values missing.

Some of the most incomplete columns are:

| Column | Approximate missing percentage |
|---|---:|
| `COMMONAREA_AVG` | 69.87% |
| `COMMONAREA_MODE` | 69.87% |
| `COMMONAREA_MEDI` | 69.87% |
| `NONLIVINGAPARTMENTS_AVG` | 69.43% |
| `NONLIVINGAPARTMENTS_MODE` | 69.43% |
| `NONLIVINGAPARTMENTS_MEDI` | 69.43% |
| `FONDKAPREMONT_MODE` | 68.39% |
| `LIVINGAPARTMENTS_AVG` | 68.35% |
| `LIVINGAPARTMENTS_MODE` | 68.35% |
| `LIVINGAPARTMENTS_MEDI` | 68.35% |
| `YEARS_BUILD_AVG` | approximately 66.5% |
| `OWN_CAR_AGE` | 65.99% |
| `LANDAREA_AVG` | approximately 59.38% |

The EDA notebook's data-quality summary reports an overall missing-cell rate of 23.91% for its 126-column in-memory training table. The corresponding rate for `application_test.csv` is 23.81%. These percentages should be interpreted with the notebook's processing state in mind; the original training CSV has 122 columns, while the in-memory EDA table includes engineered columns.

Missingness also appears in historical tables. For example, the notebook reports substantial missing values in several `bureau.csv` fields, including `AMT_CREDIT_MAX_OVERDUE` (1,124,488 missing), `AMT_ANNUITY` (1,226,791 missing), and `DAYS_ENDDATE_FACT` (633,653 missing). In `previous_application.csv`, `RATE_INTEREST_PRIMARY` and `RATE_INTEREST_PRIVILEGED` each have 1,664,263 missing values.

**Status:** Missing values have been identified in the EDA outputs. The configured preprocessing pipeline should still be inspected to verify how each feature is imputed, encoded, dropped, or otherwise transformed before model prediction.

### 3.2 Duplicate Records

The additional duplicate-check results show **zero exact duplicate rows** in each of the eight checked CSV datasets:

| Dataset | Exact duplicate rows |
|---|---:|
| `application_train.csv` | 0 |
| `application_test.csv` | 0 |
| `bureau.csv` | 0 |
| `bureau_balance.csv` | 0 |
| `previous_application.csv` | 0 |
| `POS_CASH_balance.csv` | 0 |
| `installments_payments.csv` | 0 |
| `credit_card_balance.csv` | 0 |

The checks also found zero repeated `SK_ID_CURR` values in `application_train.csv` and `application_test.csv`, consistent with one row per applicant in these application-level datasets.

Repeated `SK_ID_CURR` values in historical tables such as `bureau`, `previous_application`, `POS_CASH_balance`, `installments_payments`, and `credit_card_balance` are expected because an applicant can have multiple historical records. These counts must not be interpreted as duplicate rows or data-quality failures. `bureau_balance.csv` does not contain `SK_ID_CURR` directly; it links to bureau records using `SK_ID_BUREAU`.

**Status:** Exact duplicate rows were checked in all eight listed datasets and none were found. Application-level applicant-ID uniqueness was also checked and no duplicate IDs were found.

### 3.3 Data Types and Feature Consistency

The EDA notebook reports that the original training dataset has 122 columns: 65 `float64`, 41 `int64`, and 16 `object` columns. The target variable is included in the training table but not in the test table.

The notebook adds engineered features such as `AGE_YEARS`, `YEARS_EMPLOYED`, and `CREDIT_INCOME_RATIO` during analysis, increasing its in-memory training DataFrame from 122 to 126 columns. This is an EDA-stage change and does not, by itself, establish the number or order of features passed to the trained model.

The project documentation states that the saved model feature configuration expects 226 input features. The additional check file does not contain the feature configuration's contents, so it does not confirm that the feature list exists, has exactly 226 entries, or matches the final processed matrix.

**Status:** Source-table dimensions and data types are documented. The 226-feature model input contract remains to be verified against the actual saved feature configuration and preprocessing output.

### 3.4 Invalid or Unusual Values

The EDA notebook identifies `DAYS_EMPLOYED = 365243` as a special value. It occurs 55,374 times, representing approximately 18.01% of the training applicants. Among rows with this value, 55,352 are labelled as `Pensioner` and 22 as `Unemployed`.

The notebook replaces this value with a missing value in its in-memory `work` DataFrame before deriving employment-related features. This is a transformation performed in the notebook; the original CSV value is not thereby changed.

The notebook also applies the interquartile range (IQR) rule to `AMT_INCOME_TOTAL`. It reports IQR limits of -22,500 and 337,500, with 14,035 observations outside those limits and a maximum income value of 117,000,000. IQR outliers are unusual observations, not automatically invalid records; they should be investigated in the context of the feature and domain.

The notebook's numeric correlation analysis reports that the largest-magnitude correlations with `TARGET` among the listed features include `EXT_SOURCE_3` (-0.1789), `EXT_SOURCE_2` (-0.1605), and `EXT_SOURCE_1` (-0.1553). These are associations and should not be interpreted as causal relationships.

**Status:** The EDA notebook has identified and quantified notable unusual values and income outliers. Further domain-specific validation of numerical ranges and the preprocessing treatment of these values remains advisable.

### 3.5 Target Variable

The training target distribution reported by the EDA notebook is:

| `TARGET` | Meaning | Count | Percentage |
|---|---|---:|---:|
| 0 | No repayment difficulty recorded | 282,686 | 91.93% |
| 1 | Repayment difficulty recorded | 24,825 | 8.07% |
| **Total** | | **307,511** | **100.00%** |

The target is therefore imbalanced: the positive/default class represents approximately 8.07% of the training records. A model can achieve high accuracy by predicting the majority class too often, so accuracy should not be used as the sole evaluation metric.

**Status:** The target counts and class proportions have been checked in the EDA notebook. A separate validation of missing or unexpected target labels should be retained as a preprocessing/data-validation check.

## 4. Preprocessing

The EDA notebook performs exploratory transformations, including replacing the special `DAYS_EMPLOYED` value with a missing value and creating derived features. These operations are applied to the notebook's in-memory DataFrame.

Before relying on the model in deployment, inspect the saved preprocessing pipeline to confirm that it handles missing numerical values, categorical variables, feature ordering, and the expected 226 input features consistently during training and prediction. The checks summarized in this report do not independently verify every transformation in that saved pipeline.

## 5. Impact on Model Performance

The checks identify several issues relevant to modelling:

- Many columns contain missing values, and 41 training columns have more than 50% missing data.
- `DAYS_EMPLOYED` contains a special value that requires deliberate handling.
- `AMT_INCOME_TOTAL` has a number of IQR outliers, including very large values.
- The target is imbalanced, with approximately 8.07% positive/default cases.
- Historical tables contain multiple records per applicant by design, so row-level duplication checks must be distinguished from applicant-ID repetition.

These findings do not, by themselves, establish the model's predictive performance. The trained model should be evaluated using appropriate metrics, including precision, recall, F1-score, and ROC-AUC, alongside accuracy and a confusion matrix.

## 6. Recommendations

- Keep source CSVs unchanged and perform transformations through a reproducible preprocessing pipeline.
- Document missing-value handling for each feature, particularly columns with more than 50% missing values.
- Review the `DAYS_EMPLOYED = 365243` treatment and ensure the same transformation is applied at inference time.
- Investigate extreme income values rather than automatically deleting every IQR outlier.
- Treat repeated applicant IDs in historical tables as expected one-to-many relationships unless a table-specific key check proves otherwise.
- Verify the saved feature configuration and confirm that the final processed input has the expected 226 feature names in the correct order.
- Validate target labels and preserve the observed class distribution when designing evaluation.
- Evaluate the model with multiple metrics, especially recall and ROC-AUC, because the default class is a minority class.
- Keep the notebook outputs and generated data-quality CSV available so that findings can be reproduced.

## 7. Conclusion

The EDA and additional checks provide evidence that all eight checked CSV datasets contain zero exact duplicate rows. The application-level training and test tables also contain no repeated applicant IDs. The training data has substantial missingness in some features, a special `DAYS_EMPLOYED` value affecting approximately 18.01% of applicants, income outliers under the IQR rule, and an imbalanced target with approximately 8.07% positive/default cases.

The project documentation states that the model expects 226 input features, but the actual feature configuration and final processed feature matrix still need to be checked directly. The preprocessing pipeline should also be verified to ensure that the documented treatments are applied consistently during training and prediction.
