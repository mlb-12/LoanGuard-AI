# Data Quality Report

## 1. Introduction

This report documents the data quality considerations for the LoanGuard-AI Home Credit Default Risk prediction project. Data quality is important because missing values, duplicate records, inconsistent formats, and incorrect values can affect the reliability of machine learning predictions.

## 2. Dataset Overview

The project uses data from the Home Credit Default Risk prediction task to predict whether an applicant may experience repayment difficulties.

The dataset is processed before being used by the machine learning model. The model expects 226 input features, as specified in the project's saved model feature configuration.

## 3. Data Quality Checks

The following checks should be performed on the dataset and processed features.

### 3.1 Missing Values

Missing values should be identified in the original and processed datasets. The preprocessing pipeline should handle missing values according to its configured transformations.

**Status:** Verify using the available dataset and preprocessing pipeline before documenting the final result.

### 3.2 Duplicate Records

The dataset should be checked for duplicate rows and duplicate applicant identifiers. Duplicate records may affect model training and evaluation.

**Status:** Not yet verified.

### 3.3 Data Types and Feature Consistency

Input columns should have appropriate data types. The features supplied to the model must match the expected feature names and structure.

The saved model configuration specifies 226 input features.

**Status:** The expected feature count is documented. Verify that the final processed data matches the model's feature configuration.

### 3.4 Invalid or Unusual Values

Numerical columns such as income, credit amount, annuity, goods price, age, and employment duration should be checked for invalid or unusual values.

Values should be interpreted according to the meaning of each feature rather than automatically treating every extreme value as an error.

**Status:** Not yet verified.

### 3.5 Target Variable

The target variable should be checked for missing or invalid labels and the distribution of its classes. The class distribution is important when evaluating a loan-default prediction model.

**Status:** Verify using the training dataset.

## 4. Preprocessing

The project includes a saved preprocessing pipeline. Its transformations should be checked to confirm that missing values, categorical variables, numerical variables, and feature ordering are handled consistently during training and prediction.

## 5. Impact on Model Performance

Data quality issues can affect the reliability of model predictions. In particular, missing or inconsistent features may lead to incorrect inputs, while an imbalanced target variable may make accuracy alone an insufficient measure of performance.

The model should therefore be evaluated using metrics such as precision, recall, F1-score, and ROC-AUC, alongside accuracy.

## 6. Recommendations

- Check missing values in the original and processed datasets.
- Check duplicate records and applicant identifiers.
- Verify data types and expected feature names.
- Inspect unusual numerical values.
- Examine the target class distribution.
- Confirm that the preprocessing pipeline is consistent between training and deployment.
- Evaluate the model using multiple metrics rather than accuracy alone.

## 7. Conclusion

Data quality checks are an important part of the LoanGuard-AI workflow. The project's saved model expects 226 features, and the preprocessing pipeline is intended to prepare applicant data for prediction. The checks listed in this report must be completed and their results recorded before making definitive claims about the dataset's quality.
