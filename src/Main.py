import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(path="Data/customer-data.csv"):
    """
    Load dataset from the Data directory.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return pd.DataFrame()


def prepare_features(df):
    """
    Select numeric columns only.
    PCA cannot run on non-numeric data.
    """
    return df.select_dtypes(include=[np.number])


def get_variance_explained(df):
    """
    Fit PCA on standardized data and return explained variance ratio.
    """
    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return explained


def get_important_features(df, n_components=2):
    """
    Return the most influential feature for each principal component.
    """
    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    components = pd.DataFrame(
        pca.components_,
        columns=X.columns,
        index=[f"PC{i+1}" for i in range(n_components)]
    )

    important_features = {
        pc: components.loc[pc].abs().idxmax()
        for pc in components.index
    }

    return important_features, components


if __name__ == "__main__":
    """
    Example usage of PCA functions
    """
    data = load_data()

    explained = get_variance_explained(data)
    important_features, loadings = get_important_features(data, n_components=2)

    print("Important Features for the first 2 Principal Components:")
    for pc, feature in important_features.items():
        print(f"{pc}: {feature}")

    print("\nCumulative Variance Explained:")
    print(explained.cumsum())

    print("\nPCA Loadings (absolute importance per feature):")
    print(loadings.abs())
