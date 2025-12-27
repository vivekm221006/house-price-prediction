"""
Main script for house price prediction.
Orchestrates the complete ML pipeline: data loading, preprocessing, 
feature engineering, model training, evaluation, and visualization.
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data, get_data_info
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer


def main():
    """
    Main function to run the complete house price prediction pipeline.
    """
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("="*50)
    
    # Step 1: Load data
    print("\n[Step 1/6] Loading data...")
    X, y = load_data()
    get_data_info(X, y)
    
    # Step 2: Feature Engineering
    print("\n[Step 2/6] Engineering features...")
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.create_features(X)
    
    # Step 3: Preprocess data
    print("\n[Step 3/6] Preprocessing data...")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X_engineered, y)
    
    # Step 4: Train models
    print("\n[Step 4/6] Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train, optimize_hyperparameters=True)
    
    # Step 5: Evaluate models
    print("\n[Step 5/6] Evaluating models...")
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all_models(models, X_train, X_test, y_train, y_test)
    best_model_name = evaluator.print_results(results_df)
    evaluator.save_results(results_df, 'model_results.csv')
    
    # Step 6: Create visualizations
    print("\n[Step 6/6] Creating visualizations...")
    visualizer = Visualizer(output_dir='visualizations')
    visualizer.create_all_visualizations(X_engineered, y, X_test, y_test, models, results_df)
    
    # Summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nBest Model: {best_model_name}")
    print(f"Results saved to: model_results.csv")
    print(f"Visualizations saved to: visualizations/")
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
