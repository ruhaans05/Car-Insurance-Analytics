import numpy as np
import pandas as pd
import pytest

from transform_data import pca_feature_importance


@pytest.fixture
def sample_data():
    """
    Creates a mock dataset that mimics the actual car insurance data structure.
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [0, 2, 1, 3, 0],              # Ordinal rank (16-25, etc.)
        'income': [1, 3, 2, 3, 0],           # Ordinal rank (working, upper, etc.)
        'driving_experience': [0, 2, 1, 3, 0], # Ordinal rank (0-9y, etc.)
        'credit_score': [0.65, 0.82, 0.45, 0.91, 0.55],
        'race_encoded': [0, 1, 0, 1, 0],     # Sensitive feature
        'outcome': [0, 1, 0, 0, 1],          # Target feature
        'non_numeric_col': ['sedan', 'sedan', 'suv', 'sedan', 'suv']
    })

def test_pca_runs_on_scaled_data(sample_data):
    """
    PCA should execute and return the correct types despite different scales 
    (e.g., credit_score vs age).
    """
    to_drop, importance = pca_feature_importance(sample_data)

    assert isinstance(to_drop, list)
    assert isinstance(importance, pd.Series)
    assert not importance.empty

def test_non_numeric_columns_are_ignored(sample_data):
    """
    PCA must filter out categorical/string columns to avoid sklearn errors.
    """
    _, importance = pca_feature_importance(sample_data)

    assert 'non_numeric_col' not in importance.index

def test_excluded_columns_are_not_in_importance(sample_data):
    """
    Verify that sensitive and ID columns are explicitly removed from PCA math.
    """
    _, importance = pca_feature_importance(sample_data)

    # These should be ignored based on EXCLUDE_FROM_PCA in transform_data.py
    assert 'id' not in importance.index
    assert 'race_encoded' not in importance.index
    assert 'outcome' not in importance.index

def test_empty_dataframe_handling():
    """
    The function should return safe defaults for empty inputs.
    """
    empty_df = pd.DataFrame()
    to_drop, importance = pca_feature_importance(empty_df)

    assert to_drop == []
    assert importance.empty

def test_importance_scores_non_negative(sample_data):
    """
    Weights calculated from absolute loadings and explained variance 
    must be positive.
    """
    _, importance = pca_feature_importance(sample_data)

    assert (importance >= 0).all()

def test_pca_threshold_logic(sample_data):
    """
    Verify that if importance_threshold is set high, columns are actually added to to_drop.
    """
    # Force a very high threshold so almost everything is dropped
    to_drop, _ = pca_feature_importance(sample_data, importance_threshold=1.0)
    
    # In a small dataset, most features won't meet a 100% importance threshold
    assert len(to_drop) > 0