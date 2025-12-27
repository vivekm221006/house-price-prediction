"""
Data preprocessing module for house price prediction.
Handles missing values, feature encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Missing value imputation
    - Feature scaling
    - Train-test split
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def handle_missing_values(self, X):
        """
        Handle missing values using median imputation.
        
        Args:
            X: Features dataframe
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        print("\nHandling missing values...")
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found {missing_counts.sum()} missing values")
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print("Missing values imputed using median strategy")
            return X_imputed
        else:
            print("No missing values found")
            return X
    
    def split_data(self, X, y):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"\nSplitting data: {(1-self.test_size)*100:.0f}% train, {self.test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            tuple: Scaled X_train, X_test
        """
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, X, y):
        """
        Complete preprocessing pipeline.
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            tuple: X_train_scaled, X_test_scaled, y_train, y_test
        """
        print("\n" + "="*50)
        print("PREPROCESSING DATA")
        print("="*50)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\nPreprocessing completed!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
