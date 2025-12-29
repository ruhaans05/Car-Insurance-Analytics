import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


EXCLUDE_FROM_PCA = {"id", "race", "race_encoded", "outcome"}

def pca_feature_importance(
    df,
    variance_threshold=0.9,
    importance_threshold=0.02, 
):
    """
    Identify low-contributing numeric features using PCA.
    Standardizes data to ensure features like 'income' don't dominate 'age'.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    features = [c for c in numeric_df.columns if c.lower() not in EXCLUDE_FROM_PCA]

    if not features:
        return [], pd.Series(dtype=float)

    X = numeric_df[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    pca.fit(X_scaled)

    
    loadings = np.abs(pca.components_.T)
    importance = (loadings * pca.explained_variance_ratio_).sum(axis=1)

    importance_series = pd.Series(importance, index=features).sort_values(ascending=False)
    
    to_drop = importance_series[importance_series < importance_threshold].index.tolist()

    print(f"\nPCA Analysis: {pca.n_components_} components explain {variance_threshold*100}% of variance.")
    
    return to_drop, importance_series