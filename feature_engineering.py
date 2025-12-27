"""
Feature engineering module for house price prediction.
Creates new features from existing ones.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Creates new features to improve model performance.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
    
    def create_features(self, X):
        """
        Create engineered features from the California Housing dataset.
        
        Args:
            X: Features dataframe
            
        Returns:
            pd.DataFrame: Dataframe with original and engineered features
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        X_eng = X.copy()
        
        # Create rooms per household
        if 'AveRooms' in X.columns and 'AveBedrms' in X.columns:
            X_eng['RoomsPerBedroom'] = X['AveRooms'] / (X['AveBedrms'] + 1e-6)
            print("Created feature: RoomsPerBedroom")
        
        # Create bedrooms ratio
        if 'AveBedrms' in X.columns and 'AveRooms' in X.columns:
            X_eng['BedroomRatio'] = X['AveBedrms'] / (X['AveRooms'] + 1e-6)
            print("Created feature: BedroomRatio")
        
        # Create population per household
        if 'Population' in X.columns and 'AveOccup' in X.columns:
            X_eng['PopulationDensity'] = X['Population'] / (X['AveOccup'] + 1e-6)
            print("Created feature: PopulationDensity")
        
        # Create income categories
        if 'MedInc' in X.columns:
            X_eng['IncomeCategory'] = pd.cut(
                X['MedInc'], 
                bins=[0, 2.5, 4.5, 6.0, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)
            print("Created feature: IncomeCategory")
        
        # Create location features (interaction between latitude and longitude)
        if 'Latitude' in X.columns and 'Longitude' in X.columns:
            X_eng['LatLongInteraction'] = X['Latitude'] * X['Longitude']
            print("Created feature: LatLongInteraction")
        
        # Log transform of population (to handle skewness)
        if 'Population' in X.columns:
            X_eng['LogPopulation'] = np.log1p(X['Population'])
            print("Created feature: LogPopulation")
        
        # Total rooms per capita
        if 'AveRooms' in X.columns and 'AveOccup' in X.columns:
            X_eng['RoomsPerPerson'] = X['AveRooms'] / (X['AveOccup'] + 1e-6)
            print("Created feature: RoomsPerPerson")
        
        print(f"\nTotal features: {X_eng.shape[1]} (added {X_eng.shape[1] - X.shape[1]} new features)")
        
        return X_eng
