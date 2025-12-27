"""
Model evaluation module for house price prediction.
Evaluates models using MAE, MSE, RMSE, and R² metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """
    Evaluates regression models using various metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate a single model on both training and testing sets.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            model_name: Name of the model
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for training set
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Calculate metrics for testing set
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store results
        results = {
            'Model': model_name,
            'Train_MAE': train_mae,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_R2': train_r2,
            'Test_MAE': test_mae,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        }
        
        self.results[model_name] = results
        
        return results
    
    def evaluate_all_models(self, models, X_train, X_test, y_train, y_test):
        """
        Evaluate all models.
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        
        return results_df
    
    def print_results(self, results_df):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results_df: Results dataframe
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\n--- Training Set Performance ---")
        train_cols = ['Model', 'Train_MAE', 'Train_MSE', 'Train_RMSE', 'Train_R2']
        print(results_df[train_cols].to_string(index=False))
        
        print("\n--- Testing Set Performance ---")
        test_cols = ['Model', 'Test_MAE', 'Test_MSE', 'Test_RMSE', 'Test_R2']
        print(results_df[test_cols].to_string(index=False))
        
        # Find best model based on test R2 score
        best_model = results_df.loc[results_df['Test_R2'].idxmax()]
        
        print("\n" + "="*50)
        print("BEST MODEL")
        print("="*50)
        print(f"Model: {best_model['Model']}")
        print(f"Test MAE: {best_model['Test_MAE']:.4f}")
        print(f"Test MSE: {best_model['Test_MSE']:.4f}")
        print(f"Test RMSE: {best_model['Test_RMSE']:.4f}")
        print(f"Test R²: {best_model['Test_R2']:.4f}")
        
        return best_model['Model']
    
    def save_results(self, results_df, filename='model_results.csv'):
        """
        Save results to a CSV file.
        
        Args:
            results_df: Results dataframe
            filename: Output filename
        """
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
