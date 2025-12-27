"""
Data Preprocessing Module
Handles data cleaning, missing values, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    print("\nChecking for missing values...")
    missing_counts = data.isnull().sum()
    print(f"Missing values per column:\n{missing_counts}")
    
    if missing_counts.sum() == 0:
        print("No missing values found!")
    else:
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
                print(f"Filled missing values in {col} with median")
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_values = data[col].mode()
                if len(mode_values) > 0:
                    data[col].fillna(mode_values[0], inplace=True)
                    print(f"Filled missing values in {col} with mode")
                else:
                    # If no mode exists (all values are null), fill with a placeholder
                    data[col].fillna('Unknown', inplace=True)
                    print(f"Filled missing values in {col} with 'Unknown'")
    
    return data


def encode_categorical_features(data):
    """
    Encode categorical features using Label Encoding.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with encoded categorical features
    """
    print("\nEncoding categorical features...")
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("No categorical features found!")
        return data
    
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
        print(f"Encoded column: {col}")
    
    return data


def split_features_target(data, target_column='MedHouseVal'):
    """
    Split dataset into features and target variable.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) - features and target
    """
    print(f"\nSplitting features and target (target: {target_column})...")
    y = data[target_column]
    X = data.drop(columns=[target_column])
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Features scaled successfully!")
    
    return X_train_scaled, X_test_scaled, scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data into train ({100*(1-test_size):.0f}%) and test ({100*test_size:.0f}%) sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(data, target_column='MedHouseVal', test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Args:
        data (pd.DataFrame): Raw dataset
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("=" * 80)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Encode categorical features
    data = encode_categorical_features(data)
    
    # Split features and target
    X, y = split_features_target(data, target_column)
    
    # Split train and test
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
