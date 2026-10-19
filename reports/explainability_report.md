# Explainability Report — LoanGuard-AI

## 1. Introduction

LoanGuard-AI is a machine learning application designed to estimate the probability of repayment difficulties for loan applicants. The application uses a trained classification model to generate predictions and provides explanations to help users understand the factors influencing individual predictions.

To improve the interpretability of the model, the project integrates two Explainable Artificial Intelligence (XAI) techniques:

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

These methods provide complementary explanations of the model's predictions. They do not guarantee that a prediction is correct or establish a causal relationship between an applicant's characteristics and repayment outcomes.

## 2. Objectives

The main objectives of the explainability component are:

1. To understand how individual input features contribute to a model prediction.
2. To identify features that influence the estimated repayment-difficulty risk for an individual applicant.
3. To make model predictions easier to interpret.
4. To improve transparency by presenting explanations alongside prediction results.
5. To communicate the limitations of automated loan-risk predictions.

## 3. Explainability Techniques Used

### 3.1 SHAP (SHapley Additive exPlanations)

SHAP is a feature-attribution method based on Shapley values from cooperative game theory. It estimates the contribution of individual features to a model output relative to a reference or baseline output.

#### Implementation

The application uses the SHAP library and creates a `TreeExplainer` for the trained model.

The transformed applicant data is passed to the SHAP explainer, and a waterfall plot is generated to display the contributions of individual features.

The application displays up to 12 features in the waterfall plot.

#### Interpretation

- Features with positive SHAP values move the model output in the positive direction for the output being explained.
- Features with negative SHAP values move the model output in the negative direction.
- The magnitude of a SHAP value indicates the size of a feature's contribution relative to the other displayed contributions.
- The baseline represents the explainer's reference output, and the feature contributions explain the difference between that baseline and the output for the individual applicant.

For the binary classification model, the SHAP waterfall may display contributions in raw model-score units rather than probability percentages. Therefore, SHAP values should not automatically be interpreted as percentage changes in repayment risk.

#### Purpose

SHAP helps users investigate why the model produced a particular output for an applicant by showing the contributions attributed to individual features.

### 3.2 LIME (Local Interpretable Model-agnostic Explanations)

LIME is a local explanation technique that approximates a model's behaviour around a particular input using an interpretable surrogate model.

It generates perturbed examples around the applicant's feature values, obtains predictions for those examples, and fits a local approximation to explain the prediction.

#### Implementation

The application uses the `LimeTabularExplainer` class from the LIME library.

A sample of 500 rows from the processed training data is used as reference data for the LIME explainer. The sample contains the 226 model input features.

For each applicant:

1. The applicant's input data is transformed using the saved preprocessing pipeline.
2. The transformed features are arranged in the order expected by the trained model.
3. LIME generates a local explanation using the model's `predict_proba` function.
4. The explanation displays up to 10 features.
5. The application plots the explanation for class 1, representing repayment difficulties.

#### Interpretation

LIME shows which features support or oppose the locally approximated prediction for the selected class.

The direction and magnitude of the displayed weights indicate the influence assigned by the local surrogate model under its chosen representation and perturbation procedure.

LIME weights are not probabilities, and they should not be interpreted as causal effects. Since LIME uses a local approximation, its explanation may vary with the reference data, perturbations, and settings.

#### Purpose

LIME provides a second perspective on an individual prediction, complementing the feature-attribution explanation produced by SHAP.

## 4. Explainability Workflow

The explainability process in LoanGuard-AI follows these steps:

1. **Applicant input:** The user enters applicant information through the Streamlit interface.
2. **Prediction:** The existing prediction function estimates the probability of repayment difficulties and determines the predicted class using the configured threshold.
3. **Preprocessing:** The applicant's data is transformed using the saved preprocessing pipeline.
4. **Feature alignment:** The transformed data is aligned with the feature columns and order expected by the trained model.
5. **SHAP explanation:** SHAP calculates feature contributions and displays a waterfall plot.
6. **LIME explanation:** LIME generates a local explanation for class 1 and displays a feature-weight plot.
7. **Result presentation:** The application presents prediction results first, followed by the SHAP and LIME explanations.

## 5. Integration with the Streamlit Application

The explainability component is integrated directly into the Streamlit application.

The following resources are loaded from the project:

- `models/loan_default_model.pkl` — the trained classification model.
- `models/preprocessing_pipeline.joblib` — the saved preprocessing pipeline.
- `models/model_feature_columns.json` — the expected model feature order.
- `explainability/lime_background.pkl` — the reference sample used by LIME.

The application caches the model and explanation resources to avoid repeatedly loading and initializing them during normal Streamlit reruns.

After a prediction is generated, the application displays the estimated repayment-difficulty probability, the selected classification threshold, and the predicted class. It then generates the SHAP and LIME visualizations for the applicant.

## 6. Comparison of SHAP and LIME

| Aspect | SHAP | LIME |
|---|---|---|
| Main approach | Shapley-value-based feature attribution | Local surrogate-model approximation |
| Application in LoanGuard-AI | Uses `TreeExplainer` | Uses `LimeTabularExplainer` |
| Explanation output | Waterfall plot | Feature-weight plot |
| Features displayed | Up to 12 | Up to 10 |
| Explanation focus | Contributions to the selected model output | Local approximation of the selected class prediction |
| Main limitation | Output scale and baseline must be interpreted correctly | Local approximation and weights can be sensitive to settings and perturbations |

Using both techniques provides complementary views of individual predictions. However, agreement between two explanations does not independently prove that the prediction is correct or unbiased.

## 7. Limitations

The explainability component has the following limitations:

1. **No guarantee of prediction accuracy:** SHAP and LIME explain model behaviour; they do not measure whether the model predicts correctly.
2. **No causal interpretation:** Feature contributions do not establish that changing a feature would cause a corresponding change in repayment behaviour.
3. **Local explanation limitations:** The explanations generated for one applicant should not automatically be generalized to all applicants.
4. **Reference-data dependence:** LIME explanations depend partly on the reference sample and the perturbations generated around the applicant.
5. **Feature interpretation:** The model uses transformed features, so understanding some contributions may require examining the preprocessing and feature-engineering steps.
6. **SHAP output scale:** SHAP values may represent raw model scores rather than probabilities, depending on the model and explainer configuration.
7. **Fairness and bias:** Explainability alone does not establish that the model is fair, unbiased, or suitable for real-world lending decisions.
8. **Input limitations:** The application accepts a limited set of applicant inputs, while the model operates on a larger transformed feature set. Preprocessing may impute missing information.

## 8. Conclusion

LoanGuard-AI integrates SHAP and LIME to improve the interpretability of its loan-risk predictions. SHAP presents feature contributions to an individual model output, while LIME provides a local approximation of the model's behaviour for the repayment-difficulty class.

The two techniques complement the application's prediction results by making aspects of model behaviour easier to inspect. Their outputs should be interpreted carefully and alongside independent model evaluation, data-quality assessment, and fairness analysis.

The explainability component is intended to support understanding and investigation of model predictions. It should not be treated as a substitute for model validation or as the sole basis for consequential lending decisions.

## 9. Tools and Libraries

- Python
- Streamlit
- SHAP
- LIME
- Pandas
- NumPy
- Matplotlib
- Joblib
- Scikit-learn
- XGBoost

