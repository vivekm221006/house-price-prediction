"""
Visualization module for house price prediction.
Creates various plots to visualize data insights and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class Visualizer:
    """
    Creates visualizations for data analysis and model evaluation.
    """
    
    def __init__(self, output_dir='visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_target_distribution(self, y, save=True):
        """
        Plot the distribution of target variable.
        
        Args:
            y: Target series
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('House Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of House Prices')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: target_distribution.png")
    
    def plot_correlation_matrix(self, X, save=True):
        """
        Plot correlation matrix of features.
        
        Args:
            X: Features dataframe
            save: Whether to save the plot
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: correlation_matrix.png")
    
    def plot_feature_importance(self, model, feature_names, model_name, save=True):
        """
        Plot feature importance for linear models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            save: Whether to save the plot
        """
        if not hasattr(model, 'coef_'):
            print(f"Cannot plot feature importance for {model_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get coefficients
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        importance['Abs_Coefficient'] = importance['Coefficient'].abs()
        importance = importance.sort_values('Abs_Coefficient', ascending=False)
        
        # Plot top 15 features
        top_features = importance.head(15)
        
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        if save:
            filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: feature_importance_{model_name.replace(' ', '_').lower()}.png")
    
    def plot_predictions_vs_actual(self, y_test, predictions, model_name, save=True):
        """
        Plot predicted vs actual values.
        
        Args:
            y_test: Actual target values
            predictions: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual House Price')
        plt.ylabel('Predicted House Price')
        plt.title(f'Predicted vs Actual - {model_name}')
        plt.legend()
        plt.tight_layout()
        
        if save:
            filename = f"predictions_vs_actual_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: predictions_vs_actual_{model_name.replace(' ', '_').lower()}.png")
    
    def plot_residuals(self, y_test, predictions, model_name, save=True):
        """
        Plot residuals (errors) distribution.
        
        Args:
            y_test: Actual target values
            predictions: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        residuals = y_test - predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals scatter plot
        axes[0].scatter(predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted House Price')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residual Plot - {model_name}')
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {model_name}')
        
        plt.tight_layout()
        
        if save:
            filename = f"residuals_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: residuals_{model_name.replace(' ', '_').lower()}.png")
    
    def plot_model_comparison(self, results_df, save=True):
        """
        Plot comparison of model performances.
        
        Args:
            results_df: Results dataframe from evaluation
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['MAE', 'MSE', 'RMSE', 'R2']
        metric_names = ['Mean Absolute Error', 'Mean Squared Error', 
                       'Root Mean Squared Error', 'R² Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            train_col = f'Train_{metric}'
            test_col = f'Test_{metric}'
            
            x = range(len(results_df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], results_df[train_col], 
                  width, label='Train', alpha=0.8)
            ax.bar([i + width/2 for i in x], results_df[test_col], 
                  width, label='Test', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: model_comparison.png")
    
    def create_all_visualizations(self, X, y, X_test, y_test, models, results_df):
        """
        Create all visualizations.
        
        Args:
            X: Features dataframe (full or train)
            y: Target series (full or train)
            X_test: Test features
            y_test: Test target
            models: Dictionary of trained models
            results_df: Results dataframe
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        print()
        
        # Data visualizations
        self.plot_target_distribution(y)
        self.plot_correlation_matrix(X)
        
        # Model-specific visualizations
        for model_name, model in models.items():
            predictions = model.predict(X_test)
            
            # Feature importance
            self.plot_feature_importance(model, X_test.columns.tolist(), model_name)
            
            # Predictions vs Actual
            self.plot_predictions_vs_actual(y_test, predictions, model_name)
            
            # Residuals
            self.plot_residuals(y_test, predictions, model_name)
        
        # Model comparison
        self.plot_model_comparison(results_df)
        
        print(f"\nAll visualizations saved to '{self.output_dir}/' directory")
