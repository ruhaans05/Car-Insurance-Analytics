import numpy as np
import pandas as pd
import pytest

from transform_data import pca_feature_importance


@pytest.fixture
def sample_data():
    """Creates a mock dataset for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [25, 45, 35, 50, 23],
        'income': [50000, 100000, 75000, 120000, 45000],
        'driving_experience': [2, 20, 10, 25, 1],
        'race_encoded': [0, 1, 0, 1, 0],  # Explicitly encoded
        'non_numeric_col': ['A', 'B', 'C', 'D', 'E']
    })

def test_pca_runs_on_scaled_data(sample_data):
    """
    PCA should run without errors despite different feature scales.
    """
    to_drop, importance = pca_feature_importance(sample_data)

    assert isinstance(to_drop, list)
    assert isinstance(importance, pd.Series)


def test_non_numeric_columns_are_ignored(sample_data):
    """
    PCA should ignore non-numeric columns automatically.
    """
    _, importance = pca_feature_importance(sample_data)

    assert 'non_numeric_col' not in importance.index


def test_id_column_is_excluded(sample_data):
    """
    ID columns should never be included in PCA importance.
    """
    _, importance = pca_feature_importance(sample_data)

    assert 'id' not in importance.index


def test_empty_dataframe_handling():
    """
    Empty DataFrame should return empty results without crashing.
    """
    empty_df = pd.DataFrame()

    to_drop, importance = pca_feature_importance(empty_df)

    assert to_drop == []
    assert importance.empty


def test_importance_scores_non_negative(sample_data):
    """
    PCA importance scores should never be negative.
    """
    _, importance = pca_feature_importance(sample_data)

    assert (importance >= 0).all()


