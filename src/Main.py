import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():
    insurance_data = pd.read_csv("data.csv")
    return insurance_data




def get_variance_explained():
    insurance_data = load_data()

    X = insurance_data.select_dtypes(include="number")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
    )

    return explained

print(explained.cumsum())
