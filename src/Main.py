import pandas as pd

from redundant_columns import (encode_race, find_proxy_correlations,
                               mutual_information_scores)
from transform_data import pca_feature_importance

DATA_PATH = "Data/customer-data.csv"
OUTPUT_PATH = "Data/cleaned_customer_data.csv"

TARGET_COLUMN = "driving_experience"  # adjust if needed


def main():
    # 1️⃣ Load data
    df = pd.read_csv(DATA_PATH)
    df = encode_race(df)

    print("\n=== STEP 1: Proxy / Redundancy Checks ===")
    proxy_pairs = find_proxy_correlations(df)
    for a, b, corr in proxy_pairs:
        print(f"High correlation: {a} ↔ {b} ({corr:.2f})")

    print("\n=== STEP 2: Mutual Information vs Target ===")
    mi_scores = mutual_information_scores(df, TARGET_COLUMN)
    print(mi_scores)

    low_mi_features = mi_scores[mi_scores < 0.01].index.tolist()

    print("\n=== STEP 3: PCA Feature Contribution ===")
    pca_drop, pca_importance = pca_feature_importance(df)
    print("Low PCA contribution features:", pca_drop)

    print("\n=== STEP 4: Final Drop Decision ===")
    # Conservative rule: drop only if BOTH tests agree
    final_drop = sorted(set(low_mi_features) & set(pca_drop))

    print("Final columns to remove:", final_drop)

    cleaned_df = df.drop(columns=final_drop)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Cleaned dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
