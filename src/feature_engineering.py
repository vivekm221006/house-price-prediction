"""
Feature Engineering Module
Performs feature engineering and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns


def create_interaction_features(X):
    """
    Create interaction features between key variables.
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        pd.DataFrame: Features with interaction terms
    """
    print("\nCreating interaction features...")
    X_enhanced = X.copy()
    
    # For California Housing, create meaningful interactions
    if 'AveRooms' in X.columns and 'AveBedrms' in X.columns:
        X_enhanced['RoomsPerBedroom'] = X['AveRooms'] / (X['AveBedrms'] + 1e-5)
        print("Created feature: RoomsPerBedroom")
    
    if 'AveRooms' in X.columns and 'AveOccup' in X.columns:
        X_enhanced['RoomsPerPerson'] = X['AveRooms'] / (X['AveOccup'] + 1e-5)
        print("Created feature: RoomsPerPerson")
    
    if 'Population' in X.columns and 'AveOccup' in X.columns:
        X_enhanced['HouseholdsPerPopulation'] = 1 / (X['AveOccup'] + 1e-5)
        print("Created feature: HouseholdsPerPopulation")
    
    print(f"Original features: {X.shape[1]}, Enhanced features: {X_enhanced.shape[1]}")
    
    return X_enhanced


def select_features(X_train, X_test, y_train, k=10):
    """
    Select top k features using univariate feature selection.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        k (int): Number of top features to select
        
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_features)
    """
    print(f"\nSelecting top {k} features using univariate feature selection...")
    
    # Limit k to available features
    k = min(k, X_train.shape[1])
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    print(f"Selected features: {selected_features}")
    
    # Convert back to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    return X_train_selected, X_test_selected, selected_features


def analyze_feature_importance(X, y, output_path='outputs/plots/feature_importance.png'):
    """
    Analyze and visualize feature correlations with target.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        output_path (str): Path to save the plot
    """
    print("\nAnalyzing feature importance...")
    
    # Calculate correlations
    data_combined = pd.concat([X, y], axis=1)
    correlations = data_combined.corr()[y.name].drop(y.name).sort_values(ascending=False)
    
    print(f"Feature correlations with target:\n{correlations}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='barh', color='skyblue')
    plt.title('Feature Correlations with Target Variable')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {output_path}")
