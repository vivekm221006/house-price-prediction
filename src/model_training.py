"""
Model Training Module
Trains multiple regression models for house price prediction.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np


def train_linear_regression(X_train, y_train, random_state=42):
    """
    Train Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed
        
    Returns:
        Trained model
    """
    print("\nTraining Linear Regression...")
    np.random.seed(random_state)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Linear Regression training completed!")
    
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0, random_state=42):
    """
    Train Ridge Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha (float): Regularization strength
        random_state (int): Random seed
        
    Returns:
        Trained model
    """
    print("\nTraining Ridge Regression...")
    np.random.seed(random_state)
    
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)
    
    print(f"Ridge Regression training completed! (alpha={alpha})")
    
    return model


def train_lasso_regression(X_train, y_train, alpha=1.0, random_state=42):
    """
    Train Lasso Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        alpha (float): Regularization strength
        random_state (int): Random seed
        
    Returns:
        Trained model
    """
    print("\nTraining Lasso Regression...")
    np.random.seed(random_state)
    
    model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)
    
    print(f"Lasso Regression training completed! (alpha={alpha})")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train Random Forest Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators (int): Number of trees
        random_state (int): Random seed
        
    Returns:
        Trained model
    """
    print("\nTraining Random Forest Regressor...")
    np.random.seed(random_state)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Random Forest training completed! (n_estimators={n_estimators})")
    
    return model


def train_all_models(X_train, y_train, random_state=42):
    """
    Train all regression models.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary of trained models
    """
    print("=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    models = {}
    
    # Linear Regression
    models['Linear Regression'] = train_linear_regression(X_train, y_train, random_state)
    
    # Ridge Regression
    models['Ridge Regression'] = train_ridge_regression(X_train, y_train, alpha=1.0, random_state=random_state)
    
    # Lasso Regression
    models['Lasso Regression'] = train_lasso_regression(X_train, y_train, alpha=0.1, random_state=random_state)
    
    # Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train, n_estimators=100, random_state=random_state)
    
    print("\n" + "=" * 80)
    print(f"ALL {len(models)} MODELS TRAINED SUCCESSFULLY")
    print("=" * 80)
    
    return models


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
