"""
Data Loading Module
Loads housing dataset for price prediction.
Uses California Housing dataset (Boston Housing is deprecated).
"""

import pandas as pd
import numpy as np


def load_dataset(random_state=42, use_synthetic=False):
    """
    Load the California Housing dataset.
    If dataset cannot be fetched, generates synthetic data based on California Housing characteristics.
    
    Args:
        random_state (int): Random seed for reproducibility
        use_synthetic (bool): Force use of synthetic data
        
    Returns:
        pd.DataFrame: Complete dataset with features and target
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    if not use_synthetic:
        try:
            # Try to load California Housing dataset
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing(as_frame=True)
            data = housing.frame
            print("Dataset loaded successfully from sklearn!")
        except Exception as e:
            print(f"Unable to fetch dataset from sklearn: {e}")
            print("Generating synthetic data instead...")
            use_synthetic = True
    
    if use_synthetic:
        # Generate synthetic California Housing-like data
        n_samples = 20640
        
        # Generate features with realistic distributions
        MedInc = np.random.gamma(3, 2, n_samples)  # Median income
        HouseAge = np.random.uniform(1, 52, n_samples)  # House age
        AveRooms = np.random.gamma(6, 1, n_samples)  # Average rooms
        AveBedrms = np.random.gamma(1, 0.5, n_samples)  # Average bedrooms
        Population = np.random.gamma(500, 2, n_samples)  # Population
        AveOccup = np.random.gamma(3, 0.5, n_samples)  # Average occupancy
        Latitude = np.random.uniform(32.5, 42.0, n_samples)  # Latitude
        Longitude = np.random.uniform(-124.3, -114.3, n_samples)  # Longitude
        
        # Generate target with realistic relationships and noise
        # Base price influenced by income, rooms, age, location
        MedHouseVal = (
            0.5 * MedInc +  # Income is strong predictor
            0.01 * HouseAge +  # Age has small effect
            0.05 * AveRooms +  # Rooms have moderate effect
            -0.03 * AveBedrms +  # Too many bedrooms can reduce value
            -0.0001 * Population +  # Density effect
            -0.01 * AveOccup +  # Occupancy effect
            0.02 * Latitude +  # Location effect
            -0.01 * Longitude +  # Location effect
            np.random.normal(0, 0.7, n_samples)  # Random noise (increased)
        )
        
        # Clip values to realistic range (in $100,000s)
        MedHouseVal = np.clip(MedHouseVal, 0.5, 5.0)
        
        # Create DataFrame
        data = pd.DataFrame({
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude,
            'MedHouseVal': MedHouseVal
        })
        
        print("Synthetic dataset generated successfully!")
    
    print(f"Shape: {data.shape}")
    print(f"\nFeatures: {list(data.columns)}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nDataset info:")
    print(data.info())
    print(f"\nBasic statistics:\n{data.describe()}")
    
    return data


def save_dataset(data, filepath='data/housing_data.csv'):
    """
    Save the dataset to a CSV file.
    
    Args:
        data (pd.DataFrame): Dataset to save
        filepath (str): Path to save the dataset
    """
    data.to_csv(filepath, index=False)
    print(f"\nDataset saved to {filepath}")
