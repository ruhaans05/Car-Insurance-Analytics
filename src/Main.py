import numpy as np
import pandas as pd

# Assuming your other files are named accordingly
from redundant_columns import (find_proxy_correlations,
                               mutual_information_scores)
from transform_data import pca_feature_importance

DATA_PATH = "Data/customer-data.csv"
OUTPUT_PATH = "Data/cleaned_customer_data.csv"

# Updated Target: 'outcome' is whether they had a claim (True/False)
TARGET_COLUMN = "outcome" 

def preprocess_for_analysis(df):
    """
    Transforms strings and ranges into numeric values so math functions work.
    """
    d = df.copy()
    
    # 1. Map Ordinal Strings to Numbers
    maps = {
        'age': {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3},
        'driving_experience': {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3},
        'income': {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3},
        'education': {'none': 0, 'high school': 1, 'university': 2}
    }
    
    for col, mapping in maps.items():
        if col in d.columns:
            d[col] = d[col].map(mapping)

    # 2. Binary Encoding
    binary_cols = {
        'gender': {'female': 0, 'male': 1},
        'race': {'majority': 0, 'minority': 1},
        'vehicle_year': {'before 2015': 0, 'after 2015': 1},
        'vehicle_ownership': {False: 0, True: 1},
        'married': {False: 0, True: 1},
        'children': {False: 0, True: 1},
        'outcome': {False: 0, True: 1}
    }
    
    for col, mapping in binary_cols.items():
        if col in d.columns:
            d[col] = d[col].map(mapping)

    # 3. Drop purely non-numeric columns like 'id', 'postal_code', 'vehicle_type'
    # Postal code is numeric but mathematically meaningless as a coordinate.
    to_exclude = ['id', 'postal_code', 'vehicle_type']
    d = d.drop(columns=[c for c in to_exclude if c in d.columns])
    
    return d.dropna()

def main():
    if not pd.io.common.file_exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load and Preprocess
    raw_df = pd.read_csv(DATA_PATH)
    df = preprocess_for_analysis(raw_df)

    print("\n=== STEP 1: Proxy / Redundancy Checks ===")
    # This helps identify if 'race' is a proxy for 'credit_score' or 'income'
    proxy_pairs = find_proxy_correlations(df)
    for a, b, corr in proxy_pairs:
        print(f"High correlation: {a} ↔ {b} ({corr:.2f})")

    print("\n=== STEP 2: Mutual Information vs Target ===")
    # Measures which features actually help predict the 'outcome'
    mi_scores = mutual_information_scores(df, TARGET_COLUMN)
    print(mi_scores.sort_values(ascending=False))

    # Features that explain almost nothing about the outcome
    low_mi_features = mi_scores[mi_scores < 0.005].index.tolist()

    print("\n=== STEP 3: PCA Feature Contribution ===")
    # Identifies features that don't contribute to the dataset's variance
    pca_drop, _ = pca_feature_importance(df.drop(columns=[TARGET_COLUMN]))
    print("Low PCA contribution features:", pca_drop)

    print("\n=== STEP 4: Final Drop Decision ===")
    # We drop it if BOTH MI and PCA agree it is useless
    final_drop = sorted(set(low_mi_features) & set(pca_drop))
    
    # Specific logic for your project: Check if 'race' is in the drop list
    if 'race' in final_drop:
        print("DECISION: Race is mathematically redundant. Removing from model.")
    else:
        print("DECISION: Race shows mathematical significance. Manual policy review required.")

    print("Final columns to remove:", final_drop)

    cleaned_df = df.drop(columns=final_drop)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()