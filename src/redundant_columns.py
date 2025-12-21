import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def encode_race(df):
    """Safely encode race if categorical."""
    if "race" in df.columns and df["race"].dtype == "object":
        le = LabelEncoder()
        df["race_encoded"] = le.fit_transform(df["race"])
    return df


def find_proxy_correlations(df, threshold=0.7):
    """
    Identify highly correlated numeric features (potential proxies).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()

    proxy_pairs = [
        (i, j, corr.loc[i, j])
        for i in corr.columns
        for j in corr.columns
        if i != j and corr.loc[i, j] > threshold
    ]

    return proxy_pairs


def mutual_information_scores(df, target_col):
    """
    Compute mutual information of features vs target.
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
    target = df[target_col]

    mi = mutual_info_regression(numeric_df, target)
    return pd.Series(mi, index=numeric_df.columns).sort_values(ascending=False)
