"""
Data loading module for house price prediction.
Loads the California Housing dataset from scikit-learn.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data():
    """
    Load the California Housing dataset.
    
    Returns:
        pd.DataFrame: Features dataframe
        pd.Series: Target values (house prices)
    """
    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Get features and target
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    
    print(f"Dataset loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures: {list(X.columns)}")
    
    return X, y


def get_data_info(X, y):
    """
    Display information about the dataset.
    
    Args:
        X: Features dataframe
        y: Target series
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    print("\nFeatures Info:")
    print(X.info())
    
    print("\nStatistical Summary:")
    print(X.describe())
    
    print("\nTarget Variable Info:")
    print(f"Mean house value: ${y.mean():.2f}")
    print(f"Median house value: ${y.median():.2f}")
    print(f"Min house value: ${y.min():.2f}")
    print(f"Max house value: ${y.max():.2f}")
    print(f"Std house value: ${y.std():.2f}")
    
    print("\nMissing Values:")
    print(X.isnull().sum())
    
    return None
