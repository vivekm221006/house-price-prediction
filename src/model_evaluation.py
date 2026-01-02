"""
Model Evaluation Module
Evaluates regression models using various metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model on both training and testing sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        model_name (str): Name of the model
        
    Returns:
        dict: Evaluation metrics
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
    
    metrics = {
        'Model': model_name,
        'Train MAE': train_mae,
        'Train MSE': train_mse,
        'Train RMSE': train_rmse,
        'Train R²': train_r2,
        'Test MAE': test_mae,
        'Test MSE': test_mse,
        'Test RMSE': test_rmse,
        'Test R²': test_r2,
        'Predictions': y_test_pred
    }
    
    return metrics


def evaluate_all_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluate all trained models.
    
    Args:
        models (dict): Dictionary of trained models
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        
    Returns:
        pd.DataFrame: Evaluation metrics for all models
    """
    print("=" * 80)
    print("EVALUATING ALL MODELS")
    print("=" * 80)
    
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        
        # Store predictions separately
        predictions[model_name] = metrics.pop('Predictions')
        
        results.append(metrics)
        
        # Print metrics
        print(f"  Train MAE: {metrics['Train MAE']:.4f}")
        print(f"  Train MSE: {metrics['Train MSE']:.4f}")
        print(f"  Train RMSE: {metrics['Train RMSE']:.4f}")
        print(f"  Train R²: {metrics['Train R²']:.4f}")
        print(f"  Test MAE: {metrics['Test MAE']:.4f}")
        print(f"  Test MSE: {metrics['Test MSE']:.4f}")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  Test R²: {metrics['Test R²']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return results_df, predictions


def select_best_model(results_df, metric='Test R²', ascending=False):
    """
    Select the best performing model based on a metric.
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        metric (str): Metric to use for selection
        ascending (bool): Sort order
        
    Returns:
        str: Name of the best model
    """
    sorted_results = results_df.sort_values(by=metric, ascending=ascending)
    best_model = sorted_results.iloc[0]['Model']
    best_score = sorted_results.iloc[0][metric]
    
    print("\n" + "=" * 80)
    print("BEST MODEL SELECTION")
    print("=" * 80)
    print(f"Best model: {best_model}")
    print(f"Best {metric}: {best_score:.4f}")
    print("=" * 80)
    
    return best_model


def save_metrics(results_df, filepath='outputs/model_metrics.csv'):
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        filepath (str): Path to save the metrics
    """
    results_df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to {filepath}")
