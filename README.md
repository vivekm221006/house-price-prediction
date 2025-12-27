# House Price Prediction

A complete machine learning regression system to predict house prices using the California Housing dataset.

## Features

- **Data Loading**: Automatically loads the California Housing dataset from scikit-learn
- **Data Preprocessing**: 
  - Handles missing values using median imputation
  - Scales features using StandardScaler
  - Splits data into training (80%) and testing (20%) sets
- **Feature Engineering**: Creates 7 new engineered features including:
  - RoomsPerBedroom
  - BedroomRatio
  - PopulationDensity
  - IncomeCategory
  - LatLongInteraction
  - LogPopulation
  - RoomsPerPerson
- **Model Training**: Trains three regression models with hyperparameter optimization:
  - Linear Regression
  - Ridge Regression (with GridSearchCV)
  - Lasso Regression (with GridSearchCV)
- **Model Evaluation**: Reports comprehensive metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Visualizations**: Creates insightful plots including:
  - Target distribution
  - Correlation matrix
  - Feature importance
  - Predictions vs Actual
  - Residual plots
  - Model comparison charts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vivekm221006/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load the California Housing dataset
2. Perform feature engineering
3. Preprocess the data
4. Train all three regression models
5. Evaluate model performance
6. Generate visualizations
7. Save results to `model_results.csv`
8. Save all plots to `visualizations/` directory

## Project Structure

```
house-price-prediction/
├── main.py                    # Main orchestration script
├── data_loader.py            # Data loading utilities
├── preprocessing.py          # Data preprocessing pipeline
├── feature_engineering.py    # Feature engineering module
├── model_training.py         # Model training with hyperparameter tuning
├── evaluation.py             # Model evaluation and metrics
├── visualization.py          # Visualization creation
├── requirements.txt          # Python dependencies
├── model_results.csv         # Model evaluation results (generated)
└── visualizations/           # Generated plots (created on run)
```

## Requirements

- Python 3.7+
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2

## Results

The system trains and compares three regression models:
- **Linear Regression**: Baseline model without regularization
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization for feature selection

All models are evaluated using MAE, MSE, RMSE, and R² metrics on both training and testing sets.

## Skills Demonstrated

- Data preprocessing and cleaning
- Feature engineering
- Regression modeling
- Hyperparameter tuning using GridSearchCV
- Model evaluation and comparison
- Data visualization
- Python best practices and modular code design

## License

This project is licensed under the MIT License - see the LICENSE file for details.
