"""Shared feature engineering + preprocessing for the churn project."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

RAW_NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "HasCrCard", "IsActiveMember", "EstimatedSalary"]
RAW_CATEGORICAL = ["Geography", "Gender"]
ENGINEERED = ["BalanceSalaryRatio", "ProductDensity", "EngagementProduct",
              "AgeTenure", "ZeroBalance"]
NUMERIC = RAW_NUMERIC + ENGINEERED
DROP_COLS = ["Year", "CustomerId", "Surname", "Exited"]
RAW_INPUT_COLS = RAW_NUMERIC + RAW_CATEGORICAL


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive the engineered features required by the project brief."""
    X = X.copy()
    X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1.0)
    X["ProductDensity"] = X["NumOfProducts"] / (X["Tenure"] + 1.0)   # products per year of relationship
    X["EngagementProduct"] = X["IsActiveMember"] * X["NumOfProducts"]
    X["AgeTenure"] = X["Age"] * X["Tenure"]
    X["ZeroBalance"] = (X["Balance"] == 0).astype(int)
    return X


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RAW_CATEGORICAL)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("prep", make_preprocessor(scale)),
        ("model", estimator),
    ])


def risk_tier(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.30:
        return "Medium"
    return "Low"
