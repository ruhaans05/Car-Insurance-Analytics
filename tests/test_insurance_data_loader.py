import numpy as np
import pandas as pd
import pytest

from Main import (get_important_features, get_variance_explained,
                  prepare_features)


@pytest.fixture
def sample_data():
    """Creates a mock dataset for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [25, 45, 35, 50, 23],
        'income': [50000, 100000, 75000, 120000, 45000],
        'driving_experience': [2, 20, 10, 25, 1],
        'race': [0, 1, 0, 1, 0], # Already encoded for PCA
        'non_numeric_col': ['A', 'B', 'C', 'D', 'E']
    })

def test_prepare_features_removes_strings(sample_data):
    """Ensure non-numeric columns are dropped."""
    processed_df = prepare_features(sample_data)
    assert 'non_numeric_col' not in processed_df.columns
    assert processed_df.shape[1] == 5 # id, age, income, driving_exp, race

def test_variance_explained_sums_to_one(sample_data):
    """PCA explained variance ratio should sum to approximately 1.0."""
    explained = get_variance_explained(sample_data)
    assert np.isclose(explained.sum(), 1.0)

def test_get_important_features_output_shape(sample_data):
    """Check if the function returns the correct number of components."""
    n = 2
    important, loadings = get_important_features(sample_data, n_components=n)
    
    assert len(important) == n
    assert loadings.shape[0] == n
    assert loadings.shape[1] == prepare_features(sample_data).shape[1]

def test_pca_standardization_logic(sample_data):
    """
    Ensure PCA doesn't crash even if features have widely different scales 
    (Income vs Age).
    """
    try:
        get_important_features(sample_data)
    except ValueError as e:
        pytest.fail(f"PCA failed on scaled data: {e}")

def test_empty_dataframe_handling():
    """Ensure the code handles empty inputs gracefully."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError): # PCA usually raises ValueError on empty sets
        get_variance_explained(empty_df)