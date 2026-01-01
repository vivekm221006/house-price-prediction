"""
Main Pipeline for House Price Prediction
End-to-end machine learning pipeline for predicting house prices.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_dataset, save_dataset
from preprocessing import preprocess_data
from feature_engineering import create_interaction_features, select_features, analyze_feature_importance
from model_training import train_all_models, save_model
from model_evaluation import evaluate_all_models, select_best_model, save_metrics
from visualization import (
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_vs_target,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison
)


def main():
    """
    Main pipeline for house price prediction.
    """
    print("\n" + "=" * 80)
    print("HOUSE PRICE PREDICTION - END-TO-END ML PIPELINE")
    print("=" * 80 + "\n")
    
    # Set random seed for reproducibility
    RANDOM_STATE = 42
    
    # Create output directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)
    
    data = load_dataset(random_state=RANDOM_STATE)
    
    # Save raw dataset
    save_dataset(data, 'data/housing_data_raw.csv')
    
    # =========================================================================
    # STEP 2: Initial Data Exploration & Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DATA EXPLORATION & VISUALIZATION")
    print("=" * 80)
    
    # Correlation heatmap
    plot_correlation_heatmap(data, 'outputs/plots/correlation_heatmap.png')
    
    # Feature distributions
    plot_feature_distributions(data, output_path='outputs/plots/feature_distributions.png')
    
    # Feature vs target
    plot_feature_vs_target(data, output_path='outputs/plots/feature_vs_target.png')
    
    # =========================================================================
    # STEP 3: Data Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, 
        target_column='MedHouseVal',
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    
    # =========================================================================
    # STEP 4: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 80)
    
    # Create interaction features
    X_train_enhanced = create_interaction_features(X_train)
    X_test_enhanced = create_interaction_features(X_test)
    
    # Analyze feature importance
    analyze_feature_importance(X_train_enhanced, y_train, 'outputs/plots/feature_importance.png')
    
    # Feature selection (using all features for this dataset)
    # Uncomment if you want to limit features
    # X_train_selected, X_test_selected, selected_features = select_features(
    #     X_train_enhanced, X_test_enhanced, y_train, k=10
    # )
    
    # Use all enhanced features
    X_train_final = X_train_enhanced
    X_test_final = X_test_enhanced
    
    # =========================================================================
    # STEP 5: Model Training
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: MODEL TRAINING")
    print("=" * 80)
    
    models = train_all_models(X_train_final, y_train, random_state=RANDOM_STATE)
    
    # Save all models
    for model_name, model in models.items():
        model_filename = model_name.lower().replace(' ', '_')
        save_model(model, f'outputs/models/{model_filename}.pkl')
    
    # =========================================================================
    # STEP 6: Model Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 80)
    
    results_df, predictions = evaluate_all_models(
        models, X_train_final, y_train, X_test_final, y_test
    )
    
    # Save metrics
    save_metrics(results_df, 'outputs/model_metrics.csv')
    
    # Select best model
    best_model_name = select_best_model(results_df, metric='Test R²', ascending=False)
    
    # =========================================================================
    # STEP 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Actual vs Predicted
    plot_actual_vs_predicted(y_test, predictions, 'outputs/plots/actual_vs_predicted.png')
    
    # Residuals
    plot_residuals(y_test, predictions, 'outputs/plots/residuals.png')
    
    # Model comparison
    plot_model_comparison(results_df, 'outputs/plots/model_comparison.png')
    
    # =========================================================================
    # STEP 8: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOUTPUTS GENERATED:")
    print("  1. Data:")
    print("     - data/housing_data_raw.csv (raw dataset)")
    print("\n  2. Models:")
    for model_name in models.keys():
        model_filename = model_name.lower().replace(' ', '_')
        print(f"     - outputs/models/{model_filename}.pkl")
    print("\n  3. Metrics:")
    print("     - outputs/model_metrics.csv (evaluation metrics)")
    print("\n  4. Visualizations:")
    print("     - outputs/plots/correlation_heatmap.png")
    print("     - outputs/plots/feature_distributions.png")
    print("     - outputs/plots/feature_vs_target.png")
    print("     - outputs/plots/feature_importance.png")
    print("     - outputs/plots/actual_vs_predicted.png")
    print("     - outputs/plots/residuals.png")
    print("     - outputs/plots/model_comparison.png")
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
