"""
Visualization Module
Creates various plots for data analysis and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_correlation_heatmap(data, output_path='outputs/plots/correlation_heatmap.png'):
    """
    Create a correlation heatmap of all features.
    
    Args:
        data (pd.DataFrame): Dataset
        output_path (str): Path to save the plot
    """
    print("\nCreating correlation heatmap...")
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {output_path}")


def plot_feature_distributions(data, target_column='MedHouseVal', output_path='outputs/plots/feature_distributions.png'):
    """
    Plot distributions of key features.
    
    Args:
        data (pd.DataFrame): Dataset
        target_column (str): Name of target column
        output_path (str): Path to save the plot
    """
    print("\nCreating feature distribution plots...")
    
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to first 8 features for readability
    cols_to_plot = numerical_cols[:min(8, len(numerical_cols))]
    
    n_cols = 2
    n_rows = (len(cols_to_plot) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols_to_plot):
        axes[idx].hist(data[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(cols_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature distributions saved to {output_path}")


def plot_feature_vs_target(data, features=None, target_column='MedHouseVal', output_path='outputs/plots/feature_vs_target.png'):
    """
    Plot scatter plots of features vs target variable.
    
    Args:
        data (pd.DataFrame): Dataset
        features (list): List of features to plot (None for top 6)
        target_column (str): Name of target column
        output_path (str): Path to save the plot
    """
    print("\nCreating feature vs target plots...")
    
    if features is None:
        # Select top correlated features
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        correlations = data[numerical_cols].corrwith(data[target_column]).abs()
        features = correlations.nlargest(6).index.tolist()
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        axes[idx].scatter(data[feature], data[target_column], alpha=0.5, s=10)
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel(target_column, fontweight='bold')
        axes[idx].set_title(f'{feature} vs {target_column}')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature vs target plots saved to {output_path}")


def plot_actual_vs_predicted(y_test, predictions, output_path='outputs/plots/actual_vs_predicted.png'):
    """
    Plot actual vs predicted prices for all models.
    
    Args:
        y_test: Actual test values
        predictions (dict): Dictionary of predictions from different models
        output_path (str): Path to save the plot
    """
    print("\nCreating actual vs predicted plots...")
    
    n_models = len(predictions)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        # Scatter plot
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[idx].set_xlabel('Actual Prices', fontweight='bold')
        axes[idx].set_ylabel('Predicted Prices', fontweight='bold')
        axes[idx].set_title(f'{model_name}: Actual vs Predicted', fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Actual vs predicted plots saved to {output_path}")


def plot_residuals(y_test, predictions, output_path='outputs/plots/residuals.png'):
    """
    Plot residuals for all models.
    
    Args:
        y_test: Actual test values
        predictions (dict): Dictionary of predictions from different models
        output_path (str): Path to save the plot
    """
    print("\nCreating residual plots...")
    
    n_models = len(predictions)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        residuals = y_test - y_pred
        
        axes[idx].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('Predicted Prices', fontweight='bold')
        axes[idx].set_ylabel('Residuals', fontweight='bold')
        axes[idx].set_title(f'{model_name}: Residual Plot', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual plots saved to {output_path}")


def plot_model_comparison(results_df, output_path='outputs/plots/model_comparison.png'):
    """
    Create bar plots comparing model performance.
    
    Args:
        results_df (pd.DataFrame): Results dataframe with metrics
        output_path (str): Path to save the plot
    """
    print("\nCreating model comparison plots...")
    
    metrics = ['Test MAE', 'Test MSE', 'Test RMSE', 'Test R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Sort by metric
        sorted_df = results_df.sort_values(by=metric, ascending=(metric != 'Test R²'))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
        bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=colors)
        
        ax.set_xlabel(metric, fontweight='bold', fontsize=12)
        ax.set_ylabel('Model', fontweight='bold', fontsize=12)
        ax.set_title(f'Model Comparison: {metric}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', 
                   ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plots saved to {output_path}")
