"""
features.py - cleaning + feature engineering for the Home Credit Default Risk project.

Rules followed here (to prevent data leakage):
  * Nothing in this file ever looks at TARGET.
  * Every function works on one table at a time and uses only row-level or
    per-client statistics, so the same code can be applied to train and test.
  * Anything that must be LEARNED from data (imputation values, scaling,
    category lists, columns to drop) lives in preprocessing.py, not here.
"""
import numpy as np
import pandas as pd

DAYS_PLACEHOLDER = 365243  # fake "days" value used by Home Credit for unknown/not applicable


# ----------------------------------------------------------------- helpers
def _safe_div(a, b):
    """a / b, with division by zero and inf turned into NaN."""
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _has(df, cols):
    """True only if all required source columns exist (create features only when data is available)."""
    return all(c in df.columns for c in cols)


# ------------------------------------------------------- application table
def clean_application(df):
    """Fix the data-quality problems found in the EDA."""
    df = df.copy()
    if "DAYS_EMPLOYED" in df:
        # keep the information that the value was a placeholder, then remove it
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_PLACEHOLDER, np.nan)
    for col in ["CODE_GENDER", "ORGANIZATION_TYPE"]:
        if col in df:
            df[col] = df[col].replace("XNA", np.nan)
    return df


def add_application_features(df):
    """Row-level financial features (safe to compute before the train/valid split)."""
    df = df.copy()

    if _has(df, ["DAYS_BIRTH"]):
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    if _has(df, ["DAYS_EMPLOYED"]):
        df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365            # employment duration
    if _has(df, ["DAYS_EMPLOYED", "DAYS_BIRTH"]):
        df["EMPLOYED_TO_AGE_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    if _has(df, ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
        df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    if _has(df, ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
        df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    if _has(df, ["AMT_ANNUITY", "AMT_CREDIT"]):
        df["CREDIT_TERM"] = _safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])   # ~ number of payments
    if _has(df, ["AMT_GOODS_PRICE", "AMT_CREDIT"]):
        df["GOODS_CREDIT_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    if _has(df, ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]):
        df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df]
    if len(ext) >= 2:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext].min(axis=1)
        df["EXT_SOURCE_PROD"] = df[ext].prod(axis=1, skipna=False)

    return df.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------- history tables
# Every function returns ONE ROW PER SK_ID_CURR.

def aggregate_bureau(bureau, bureau_bal=None):
    """Credits from other institutions (+ optional monthly status from bureau_balance)."""
    b = bureau.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["IS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = b.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
        BUREAU_OVERDUE_SHARE=("IS_OVERDUE", "mean"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    )
    agg["BUREAU_DEBT_CREDIT_RATIO"] = _safe_div(agg["BUREAU_DEBT_SUM"], agg["BUREAU_CREDIT_SUM_SUM"])

    if bureau_bal is not None:
        bb = bureau_bal.copy()
        bb["BB_LATE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)  # DPD buckets
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS=("MONTHS_BALANCE", "count"), BB_LATE_SHARE=("BB_LATE", "mean"))
        link = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(bb_agg, on="SK_ID_BUREAU", how="left")
        bb_client = link.groupby("SK_ID_CURR").agg(
            BUREAU_BB_LATE_SHARE_MEAN=("BB_LATE_SHARE", "mean"),
            BUREAU_BB_MONTHS_SUM=("BB_MONTHS", "sum"))
        agg = agg.join(bb_client)
    return agg.reset_index()


def aggregate_previous(prev):
    """Previous applications at Home Credit."""
    p = prev.copy()
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        if c in p:
            p[c] = p[c].replace(DAYS_PLACEHOLDER, np.nan)   # same placeholder exists here
    p["APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["APP_CREDIT_RATIO"] = _safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])

    agg = p.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),                       # previous application count
        PREV_APPROVED_SHARE=("APPROVED", "mean"),
        PREV_REFUSED_SHARE=("REFUSED", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),              # average previous credit amount
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    )
    return agg.reset_index()


def aggregate_installments(inst):
    """Instalment payment behaviour."""
    i = inst.copy()
    i["DAYS_LATE"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]    # >0 = late
    i["IS_LATE"] = (i["DAYS_LATE"] > 0).astype(int)
    i["PAYMENT_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]          # >0 = underpaid
    i["UNDERPAID"] = (i["PAYMENT_DIFF"] > 0).astype(int)
    i["PAYMENT_RATIO"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])

    agg = i.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "count"),
        INST_LATE_SHARE=("IS_LATE", "mean"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_UNDERPAID_SHARE=("UNDERPAID", "mean"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_PAYMENT_DIFF_SUM=("PAYMENT_DIFF", "sum"),
    )
    return agg.reset_index()


def aggregate_pos(pos):
    """POS / cash loan monthly snapshots."""
    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_INSTAL_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    )
    return agg.reset_index()


def aggregate_credit_card(cc):
    """Credit card monthly snapshots (sparsest table)."""
    c = cc.copy()
    c["UTILIZATION"] = _safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    agg = c.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
    )
    return agg.reset_index()


# ------------------------------------------------------------- one entry point
def _merge_history(df, agg, count_col, name):
    """Left-join an aggregate; add a HAS_<name>_HISTORY flag; 'count' = 0 when no history."""
    df = df.merge(agg, on="SK_ID_CURR", how="left")
    df[f"HAS_{name}_HISTORY"] = df[count_col].notna().astype(int)
    df[count_col] = df[count_col].fillna(0)
    return df


def build_features(app, bureau=None, bureau_bal=None, prev=None, pos=None, inst=None, cc=None):
    """
    Clean the application table, add its features, then attach per-client aggregates
    from every history table that was supplied. Works for train AND test.
    Returns one row per application (same row order as `app`).
    """
    df = add_application_features(clean_application(app))
    n_before = len(df)

    if bureau is not None:
        df = _merge_history(df, aggregate_bureau(bureau, bureau_bal), "BUREAU_COUNT", "BUREAU")
    if prev is not None:
        df = _merge_history(df, aggregate_previous(prev), "PREV_COUNT", "PREV")
    if inst is not None:
        df = _merge_history(df, aggregate_installments(inst), "INST_COUNT", "INST")
    if pos is not None:
        df = _merge_history(df, aggregate_pos(pos), "POS_COUNT", "POS")
    if cc is not None:
        df = _merge_history(df, aggregate_credit_card(cc), "CC_COUNT", "CC")

    # extra cross features (only when both sides exist)
    if _has(df, ["PREV_AMT_CREDIT_MEAN", "AMT_CREDIT"]):
        df["CREDIT_TO_PREV_CREDIT_RATIO"] = _safe_div(df["AMT_CREDIT"], df["PREV_AMT_CREDIT_MEAN"])
    if _has(df, ["BUREAU_DEBT_SUM", "AMT_INCOME_TOTAL"]):
        df["BUREAU_DEBT_INCOME_RATIO"] = _safe_div(df["BUREAU_DEBT_SUM"], df["AMT_INCOME_TOTAL"])

    assert len(df) == n_before, "a merge duplicated rows - aggregates must be one row per SK_ID_CURR"
    return df.replace([np.inf, -np.inf], np.nan)
