from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


RANDOM_STATE = 42
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def evaluate(name, model, X_valid, y_valid):
    """Evaluate a fitted binary classifier at its default threshold."""
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, predictions),
        "Precision": precision_score(y_valid, predictions, zero_division=0),
        "Recall": recall_score(y_valid, predictions, zero_division=0),
        "F1-score": f1_score(y_valid, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_valid, probabilities),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train, compare, and save Member 3 loan classification models."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--models", default="models")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--skip-xgboost", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.outputs)
    model_dir = Path(args.models)
    report_dir = Path(args.reports)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_processed.pkl"
    valid_path = output_dir / "valid_processed.pkl"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {valid_path}. "
            "Run the script from the project root or provide the correct paths."
        )

    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)

    if "TARGET" not in train_df.columns or "TARGET" not in valid_df.columns:
        raise ValueError("TARGET column is missing from processed train/validation data.")

    X_train = train_df.drop(columns=["TARGET"])
    y_train = train_df["TARGET"].astype("int8")
    X_valid = valid_df.drop(columns=["TARGET"])
    y_valid = valid_df["TARGET"].astype("int8")

    if "SK_ID_CURR" in X_train.columns or "SK_ID_CURR" in X_valid.columns:
        raise ValueError("SK_ID_CURR must not be used as a model feature.")
    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Training and validation feature columns/order do not match.")
    if X_train.isna().any().any() or X_valid.isna().any().any():
        raise ValueError("Processed data contains missing values; inspect preprocessing.")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Training TARGET must contain both classes 0 and 1.")
    if not set(pd.unique(y_valid)).issubset({0, 1}):
        raise ValueError("Validation TARGET contains values other than 0 and 1.")

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print(f"Training positive rate: {y_train.mean():.4%}")
    print(f"Validation positive rate: {y_valid.mean():.4%}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }

    if not args.skip_xgboost:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=2,
            )
        except ImportError:
            raise ImportError(
                "XGBoost is required for this project. Install it with "
                "`pip install xgboost`, or explicitly use --skip-xgboost."
            )

    fitted_models = {}
    result_rows = []

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        fitted_models[name] = model
        row = evaluate(name, model, X_valid, y_valid)
        row["Training time (seconds)"] = round(training_seconds, 2)
        result_rows.append(row)
        print(row)

    print("\nTuning Logistic Regression with 3-fold stratified GridSearchCV ...")
    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=3, shuffle=True, random_state=RANDOM_STATE
        ),
        n_jobs=2,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    tuning_start = time.perf_counter()
    search.fit(X_train, y_train)
    tuning_seconds = time.perf_counter() - tuning_start

    tuned_name = "Logistic Regression (Tuned)"
    fitted_models[tuned_name] = search.best_estimator_
    tuned_row = evaluate(tuned_name, search.best_estimator_, X_valid, y_valid)
    tuned_row["Training time (seconds)"] = round(tuning_seconds, 2)
    result_rows.append(tuned_row)

    results = (
        pd.DataFrame(result_rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    results.to_csv(report_dir / "baseline_model_comparison.csv", index=False)

    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results.to_csv(
        report_dir / "logistic_regression_grid_search.csv", index=False
    )
    # Concise tuning summary for the final report.
    tuning_summary = tuning_results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_train_score",
        ]
    ].copy()
    tuning_summary.insert(0, "Model", "Logistic Regression")
    tuning_summary.to_csv(report_dir / "tuning_results.csv", index=False)

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print("\nBest Logistic Regression parameters:", search.best_params_)
    print(f"Best mean CV ROC-AUC: {search.best_score_:.4f}")

    best_name = str(results.iloc[0]["Model"])
    best_model = fitted_models[best_name]

    model_path = model_dir / "loan_default_model.pkl"
    joblib.dump(best_model, model_path)

    feature_schema_path = model_dir / "model_feature_columns.json"
    feature_schema_path.write_text(
        json.dumps(list(X_train.columns), indent=2), encoding="utf-8"
    )

    # Evaluate and save the selected model at its default threshold (0.50).
    default_predictions = best_model.predict(X_valid)
    default_probabilities = best_model.predict_proba(X_valid)[:, 1]

    report = classification_report(
        y_valid,
        default_predictions,
        target_names=["No Difficulty (0)", "Difficulty (1)"],
        digits=4,
        zero_division=0,
    )
    (report_dir / "xgboost_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    (report_dir / "best_model_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    default_cm = confusion_matrix(y_valid, default_predictions, labels=[0, 1])
    pd.DataFrame(
        default_cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(report_dir / "xgboost_confusion_matrix.csv")

    # Explore thresholds on validation probabilities. This is exploratory;
    # use a separate labelled test set for final unbiased evaluation.
    threshold_rows = []
    for threshold in THRESHOLDS:
        predictions = (default_probabilities >= threshold).astype("int8")
        tn, fp, fn, tp = confusion_matrix(
            y_valid, predictions, labels=[0, 1]
        ).ravel()

        threshold_rows.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_valid, predictions),
                "Precision": precision_score(
                    y_valid, predictions, zero_division=0
                ),
                "Recall": recall_score(y_valid, predictions, zero_division=0),
                "F1-score": f1_score(y_valid, predictions, zero_division=0),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(report_dir / "xgboost_threshold_analysis.csv", index=False)

    # Choose the tested threshold with the highest validation F1-score.
    best_threshold_row = threshold_df.loc[threshold_df["F1-score"].idxmax()]
    selected_threshold = float(best_threshold_row["Threshold"])

    threshold_config = {
        "model": best_name,
        "threshold": selected_threshold,
        "selection_metric": "F1-score",
        "selection_method": (
            "Highest F1-score among the predefined thresholds evaluated "
            "on the validation set"
        ),
        "thresholds_tested": THRESHOLDS,
        "note": (
            "Threshold selection used validation data. Evaluate once on a "
            "separate labelled test set for an unbiased final estimate."
        ),
    }
    (model_dir / "selected_threshold.json").write_text(
        json.dumps(threshold_config, indent=2), encoding="utf-8"
    )

    selected_predictions = (
        default_probabilities >= selected_threshold
    ).astype("int8")
    selected_cm = confusion_matrix(
        y_valid, selected_predictions, labels=[0, 1]
    )
    pd.DataFrame(
        selected_cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(
        report_dir / "xgboost_confusion_matrix_threshold_015.csv"
        if selected_threshold == 0.15
        else report_dir / "best_model_confusion_matrix_selected_threshold.csv"
    )

    # Verify model serialization and schema.
    reloaded_model = joblib.load(model_path)
    assert np.allclose(
        best_model.predict_proba(X_valid.iloc[:10])[:, 1],
        reloaded_model.predict_proba(X_valid.iloc[:10])[:, 1],
    )

    print(f"\nSelected model by validation ROC-AUC: {best_name}")
    print(f"Validation ROC-AUC: {results.iloc[0]['ROC-AUC']:.4f}")
    print(f"Selected threshold by validation F1-score: {selected_threshold:.2f}")
    print("Saved model:", model_path.resolve())
    print("Saved feature schema:", feature_schema_path.resolve())
    print("Saved threshold config:", (model_dir / "selected_threshold.json").resolve())
    print("Saved comparison, tuning, classification, and threshold reports.")
    print("Model reload check passed.")
    print("\nClassification report at threshold 0.50:\n", report)


if __name__ == "__main__":
    main()
