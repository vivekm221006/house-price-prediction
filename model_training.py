"""
Model training module for house price prediction.
Trains multiple regression models: Linear, Ridge, and Lasso.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import time


class ModelTrainer:
    """
    Trains and manages multiple regression models.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.best_models = {}
        
    def train_linear_regression(self, X_train, y_train):
        """
        Train a Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print("\nTraining Linear Regression...")
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"Linear Regression trained in {elapsed_time:.2f} seconds")
        
        self.models['Linear Regression'] = model
        return model
    
    def train_ridge_regression(self, X_train, y_train, optimize=True):
        """
        Train a Ridge Regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            optimize: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        print("\nTraining Ridge Regression...")
        start_time = time.time()
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            
            ridge = Ridge()
            grid_search = GridSearchCV(
                ridge, 
                param_grid, 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best alpha: {grid_search.best_params_['alpha']}")
            self.best_models['Ridge'] = grid_search.best_params_
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"Ridge Regression trained in {elapsed_time:.2f} seconds")
        
        self.models['Ridge Regression'] = model
        return model
    
    def train_lasso_regression(self, X_train, y_train, optimize=True):
        """
        Train a Lasso Regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            optimize: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        print("\nTraining Lasso Regression...")
        start_time = time.time()
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            }
            
            lasso = Lasso(max_iter=10000)
            grid_search = GridSearchCV(
                lasso, 
                param_grid, 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best alpha: {grid_search.best_params_['alpha']}")
            self.best_models['Lasso'] = grid_search.best_params_
        else:
            model = Lasso(alpha=1.0, max_iter=10000)
            model.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"Lasso Regression trained in {elapsed_time:.2f} seconds")
        
        self.models['Lasso Regression'] = model
        return model
    
    def train_all_models(self, X_train, y_train, optimize_hyperparameters=True):
        """
        Train all regression models.
        
        Args:
            X_train: Training features
            y_train: Training target
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Train Linear Regression
        self.train_linear_regression(X_train, y_train)
        
        # Train Ridge Regression
        self.train_ridge_regression(X_train, y_train, optimize=optimize_hyperparameters)
        
        # Train Lasso Regression
        self.train_lasso_regression(X_train, y_train, optimize=optimize_hyperparameters)
        
        print("\n" + "="*50)
        print(f"All {len(self.models)} models trained successfully!")
        print("="*50)
        
        return self.models
