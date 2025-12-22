import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def encode_race(df):
    """
    Safely encode race if categorical. 
    Note: Your Main.py now handles most mapping, but this remains as a safety net.
    """
    d = df.copy()
    if "race" in d.columns and d["race"].dtype == "object":
        le = LabelEncoder()
        d["race"] = le.fit_transform(d["race"])
    return d

def find_proxy_correlations(df, threshold=0.7):
    """
    Identify highly correlated numeric features (potential proxies).
    Returns a unique list of pairs to avoid A-B and B-A duplication.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()
    
    # Use the upper triangle of the correlation matrix to avoid duplicates (A-B vs B-A)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    proxy_pairs = []
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                proxy_pairs.append((row, col, upper.loc[row, col]))
                
    return proxy_pairs

def mutual_information_scores(df, target_col):
    """
    Compute mutual information of features vs target.
    Handles the case where the target column might not be numeric yet.
    """
    # 1. Ensure we have a numeric copy
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    if target_col not in numeric_df.columns:
        # If the target is still 'True/False', convert to 0/1
        target = df[target_col].astype(int)
    else:
        target = numeric_df[target_col]
        numeric_df = numeric_df.drop(columns=[target_col])

    # 2. Mutual Info cannot handle NaNs
    if numeric_df.isnull().values.any():
        numeric_df = numeric_df.fillna(0)
        target = target.loc[numeric_df.index]

    # 3. Calculate MI
    # discrete_features='auto' is smart enough to handle encoded ranks vs floats
    mi = mutual_info_regression(numeric_df, target, random_state=42)
    
    return pd.Series(mi, index=numeric_df.columns).sort_values(ascending=False)