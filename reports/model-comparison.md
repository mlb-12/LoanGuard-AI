# Model Comparison Report — LoanGuard-AI

## 1. Introduction

The purpose of this report is to compare the performance of different machine learning classification models developed for the LoanGuard-AI project. The models predict whether a loan applicant may experience repayment difficulties.

Four model configurations were evaluated:

- XGBoost
- Logistic Regression (baseline)
- Logistic Regression (tuned)
- Random Forest

The models were compared using accuracy, precision, recall, F1-score, ROC-AUC, and training time where available.

## 2. Evaluation Metrics

The following evaluation metrics were used:

- **Accuracy:** The proportion of all predictions that were correct.
- **Precision:** The proportion of applicants predicted to experience repayment difficulties who actually experienced them.
- **Recall:** The proportion of actual repayment-difficulty cases correctly identified by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well a model ranks positive cases above negative cases across classification thresholds.
- **Training time:** The time taken to train a model, measured in seconds.

Accuracy should not be interpreted in isolation because a high proportion of non-default cases in the dataset can result in high accuracy even when a model identifies very few actual repayment-difficulty cases.

## 3. Model Comparison Results

The following table presents the results provided by the model comparison experiment.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9202 | 0.6318 | 0.0280 | 0.0536 | 0.7802 | 21.95 |
| Logistic Regression (tuned) | 0.9198 | 0.5792 | 0.0236 | 0.0453 | 0.7677 | Not provided |
| Logistic Regression (baseline) | 0.9198 | 0.5805 | 0.0240 | 0.0460 | 0.7676 | 7.21 |
| Random Forest | 0.9193 | 0.5000 | 0.0010 | 0.0020 | 0.7591 | 70.74 |

*Note: The metrics are reported as fractions between 0 and 1. The training time for tuned Logistic Regression was not provided in the supplied results.*

## 4. Analysis of Results

### 4.1 Accuracy

XGBoost achieved the highest accuracy of approximately 92.02%, followed closely by tuned and baseline Logistic Regression at approximately 91.98%.

Random Forest achieved approximately 91.93% accuracy.

The differences in accuracy between the models are small. Therefore, accuracy alone is insufficient to determine which model is most suitable for identifying repayment-difficulty cases.

### 4.2 Precision

XGBoost achieved the highest precision at approximately 63.18%. This indicates that, among the cases it predicted as repayment-difficulty cases, approximately 63.18% were positive cases in the evaluated data.

Tuned Logistic Regression achieved approximately 57.92% precision, while baseline Logistic Regression achieved approximately 58.05%. Random Forest achieved 50.00% precision.

XGBoost performed best among the compared models on this metric.

### 4.3 Recall

XGBoost achieved recall of approximately 2.80%, the highest among the four evaluated configurations.

Tuned Logistic Regression achieved approximately 2.36% recall, while baseline Logistic Regression achieved approximately 2.40%. Random Forest achieved approximately 0.10%.

These recall values are very low. In particular, XGBoost identified only a small fraction of the actual repayment-difficulty cases at the evaluated classification threshold.

This is an important limitation because missing actual repayment-difficulty cases may reduce the usefulness of the model for risk-screening purposes.

### 4.4 F1-score

XGBoost achieved the highest F1-score of approximately 0.0536, followed by baseline Logistic Regression at approximately 0.0460 and tuned Logistic Regression at approximately 0.0453.

Random Forest achieved an F1-score of approximately 0.0020.

Although XGBoost performed best on F1-score, its low recall indicates that its overall identification of positive cases remains weak at the evaluated threshold.

### 4.5 ROC-AUC

XGBoost achieved the highest ROC-AUC of approximately 0.7802.

Tuned Logistic Regression achieved approximately 0.7677, baseline Logistic Regression approximately 0.7676, and Random Forest approximately 0.7591.

XGBoost therefore showed the strongest ability among the tested models to rank repayment-difficulty cases above non-difficulty cases across different classification thresholds.

ROC-AUC measures ranking performance rather than the correctness of predictions at one particular threshold.

### 4.6 Training Time

Among the models with reported training times:

- Baseline Logistic Regression took 7.21 seconds.
- XGBoost took 21.95 seconds.
- Random Forest took 70.74 seconds.

Baseline Logistic Regression trained fastest, while Random Forest took the longest among these three configurations. Training time for tuned Logistic Regression was not supplied.

XGBoost required more training time than baseline Logistic Regression but achieved better ROC-AUC, precision, recall, and F1-score in the reported comparison.

## 5. Selected Model

Based on the supplied comparison results, XGBoost is the strongest overall candidate among the four evaluated configurations because it achieved:

- The highest accuracy.
- The highest precision.
- The highest recall.
- The highest F1-score.
- The highest ROC-AUC.

The model was also saved as `loan_default_model.pkl` for use by the LoanGuard-AI application.

However, the low recall is a substantial limitation. Selecting XGBoost based on these results does not mean that its current classification decisions are sufficiently reliable for real-world lending.

The model should undergo further evaluation and threshold assessment before being considered for consequential decisions.

## 6. Recommendations for Improvement

The following steps are recommended:

1. **Investigate class imbalance:** Examine the distribution of repayment-difficulty and non-difficulty cases in the training and validation datasets.
2. **Evaluate the classification threshold:** Compare precision, recall, F1-score, and the confusion matrix at multiple thresholds. A lower threshold may identify more positive cases, although it may also increase false positives.
3. **Use a separate test dataset:** Evaluate the final selected model on labelled test data that was not used for training or threshold selection.
4. **Examine the confusion matrix:** Determine how many actual repayment-difficulty cases are missed and how many non-difficulty cases are incorrectly flagged.
5. **Assess probability calibration:** Check whether predicted probabilities correspond reasonably well to observed outcome frequencies.
6. **Perform fairness and data-quality checks:** Investigate whether model errors differ across relevant applicant groups and whether the features and preprocessing are appropriate.

Threshold selection should be guided by the intended use of the application and the relative consequences of false positives and false negatives.

## 7. Conclusion

The model comparison indicates that XGBoost performed best among the four tested configurations on the supplied evaluation metrics, achieving a ROC-AUC of approximately 0.7802 and an accuracy of approximately 92.02%.

Nevertheless, the recall of approximately 2.80% and F1-score of approximately 0.0536 reveal that the model identifies very few actual repayment-difficulty cases at the evaluated threshold. The high accuracy should therefore not be treated as sufficient evidence of effective default detection.

XGBoost is a reasonable candidate for further investigation, but additional threshold analysis, independent test evaluation, and fairness assessment are necessary before the model can be considered suitable for real-world lending decisions.

LoanGuard-AI is intended as an experimental prediction and explainability application, not as an autonomous loan approval or rejection system.