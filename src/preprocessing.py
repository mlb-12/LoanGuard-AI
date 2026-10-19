"""
preprocessing.py - reusable, leakage-safe preprocessing pipeline.

Everything that is LEARNED from data (columns to drop, imputation medians,
scaling mean/std, category lists) is learned inside a scikit-learn Pipeline,
so calling pipe.fit(X_train) learns from the training rows only and
pipe.transform(...) simply applies those learned values to validation/test data.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import build_features

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose share of missing values in the TRAINING data exceeds `threshold`."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        miss = X.isna().mean()
        self.cols_to_drop_ = miss[miss > self.threshold].index.tolist()
        self.keep_cols_ = [c for c in X.columns if c not in self.cols_to_drop_]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_cols_]


def split_xy(df, target=TARGET, id_col=ID_COL):
    """Separate features from the target and remove the ID (an ID is not a feature)."""
    y = df[target] if target in df.columns else None
    X = df.drop(columns=[c for c in (target, id_col) if c in df.columns])
    return X, y


def train_valid_split(X, y, test_size=0.2, random_state=42):
    """Stratified split so both parts keep the ~8% default rate."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(scale=True, max_missing=0.6, min_frequency=0.01):
    """
    scale=True  -> for linear / distance based models (logistic regression, NN, SVM)
    scale=False -> for tree models (RandomForest, XGBoost, LightGBM); scaling is harmless but not needed
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frequency, sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("drop_missing", HighMissingDropper(max_missing)),
                     ("prep", column_transformer)])


def get_feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def transform_to_frame(pipe, X):
    """Apply a FITTED pipeline and return a DataFrame with proper column names and index."""
    return pd.DataFrame(pipe.transform(X), columns=get_feature_names(pipe), index=X.index)


def save_pipeline(pipe, path):
    joblib.dump(pipe, path)


def load_pipeline(path):
    return joblib.load(path)


def align_to_pipeline(pipe, X):
    """Give X exactly the raw columns the fitted pipeline expects.
    Columns that are not supplied (e.g. credit history of a brand-new applicant) become NaN
    and are then filled by the pipeline's imputers."""
    keep = pipe.named_steps["drop_missing"].keep_cols_
    num_cols, cat_cols = (pipe.named_steps["prep"].transformers_[0][2],
                          pipe.named_steps["prep"].transformers_[1][2])
    X = X.reindex(columns=keep)
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    for c in cat_cols:
        X[c] = X[c].astype(object)
    return X


def transform_new_applicants(pipe, raw):
    """
    Turn raw applicant data (a dict for ONE applicant, or a DataFrame) into model-ready features.
    Uses the same cleaning + feature code as training. Works even when only the main application
    fields are known (history columns are then missing and get imputed).
    """
    df = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()
    df = build_features(df)
    X, _ = split_xy(df)
    return transform_to_frame(pipe, align_to_pipeline(pipe, X))
