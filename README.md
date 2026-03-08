# House Price Prediction & Carbon Footprint Analyzer

An end-to-end machine learning project featuring house price prediction and an AI-based carbon footprint analyzer for organizations.

## Project Overview

This project demonstrates complete machine learning pipelines including:
1. **House Price Prediction** using the California Housing dataset with data preprocessing, feature engineering, multiple regression models, and comprehensive evaluation.
2. **AI-Based Carbon Footprint Analyzer** that helps organizations measure, analyze, and reduce their carbon emissions using ML-powered classification and recommendations.

WEBSITE LINK: https://house-price-prediction-v9tf6xoigyg2y2qfxekkwh.streamlit.app/ (TRY NOW)

## Features

### 🏠 House Price Prediction
- **Data Loading**: Loads the California Housing dataset (Boston Housing is deprecated)
- **Data Preprocessing**:
  - Handle missing values
  - Encode categorical features
  - Scale numerical features
- **Feature Engineering**:
  - Create interaction features
  - Feature selection using univariate methods
  - Feature importance analysis
- **Model Training**: Multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
- **Model Evaluation**: Comprehensive metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Visualizations**:
  - Correlation heatmap
  - Feature distributions
  - Feature vs price relationships
  - Feature importance
  - Actual vs predicted prices
  - Residual plots
  - Model comparison

### 🌍 AI-Based Carbon Footprint Analyzer
- **Emission Categories**:
  - ⚡ Electricity consumption
  - ✈️ Business travel (air, car, train)
  - 🖥️ Data centers & cloud usage
  - ⛽ Fuel consumption (diesel, gasoline)
  - 🏢 Office operations (employees, waste)
- **AI-Powered Analysis**:
  - Random Forest classifier for emission tier classification (Low/Moderate/High/Very High)
  - ML-based prediction of achievable emission reduction percentage
  - Prioritized recommendations based on emission profile
- **Interactive Dashboard**:
  - Streamlit UI with real-time emission calculations
  - Pie chart and bar chart visualizations
  - Detailed breakdown table with category shares
  - Emission factors reference

## Project Structure

```
house-price-prediction/
├── data/                          # Data directory
│   └── housing_data_raw.csv       # Raw dataset
├── outputs/                       # Output directory
│   ├── models/                    # Trained models
│   │   ├── linear_regression.pkl
│   │   ├── ridge_regression.pkl
│   │   ├── lasso_regression.pkl
│   │   └── random_forest.pkl
│   ├── plots/                     # Visualizations
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions.png
│   │   ├── feature_vs_target.png
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals.png
│   │   └── model_comparison.png
│   └── model_metrics.csv          # Evaluation metrics
├── src/                           # Source code
│   ├── data_loader.py             # Data loading module
│   ├── preprocessing.py           # Data preprocessing module
│   ├── feature_engineering.py     # Feature engineering module
│   ├── model_training.py          # Model training module
│   ├── model_evaluation.py        # Model evaluation module
│   ├── visualization.py           # Visualization module
│   └── carbon_footprint.py        # Carbon footprint analysis module
├── tests/                         # Test suite
│   └── test_carbon_footprint.py   # Carbon footprint tests
├── app.py                         # House price prediction pipeline
├── streamlit_app.py               # House price Streamlit app
├── carbon_analyzer_app.py         # Carbon footprint Streamlit app
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Tech Stack

- **Python 3.7+**
- **Libraries**:
  - NumPy - Numerical computing
  - Pandas - Data manipulation
  - Scikit-learn - Machine learning
  - Matplotlib - Plotting
  - Seaborn - Statistical visualizations
  - Streamlit - Interactive web applications

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

### House Price Prediction Pipeline

Run the complete ML pipeline:

```bash
python app.py
```

This will:
1. Load the California Housing dataset
2. Perform data preprocessing and cleaning
3. Create engineered features
4. Train all regression models
5. Evaluate model performance
6. Generate visualizations
7. Save all outputs to the `outputs/` directory

### House Price Streamlit App

```bash
streamlit run streamlit_app.py
```

### Carbon Footprint Analyzer

```bash
streamlit run carbon_analyzer_app.py
```

This will launch an interactive dashboard where you can:
1. Enter your organization's data (electricity, travel, servers, fuel, employees)
2. Get AI-powered emission analysis with tier classification
3. View emission breakdown charts and tables
4. Receive prioritized reduction recommendations
5. See estimated reduction potential

### Running Tests

```bash
python -m pytest tests/ -v
```

## Outputs

After running the pipeline, you'll find:

1. **Data**: `data/housing_data_raw.csv` - Raw dataset
2. **Models**: `outputs/models/*.pkl` - Trained models (saved as pickle files)
3. **Metrics**: `outputs/model_metrics.csv` - Performance metrics for all models
4. **Visualizations**: `outputs/plots/*.png` - All generated plots

## Model Performance

The pipeline trains and compares four regression models:

- **Linear Regression**: Baseline linear model
- **Ridge Regression**: Linear model with L2 regularization
- **Lasso Regression**: Linear model with L1 regularization
- **Random Forest**: Ensemble tree-based model

All models are evaluated using MAE, MSE, RMSE, and R² Score on both training and testing sets.

## Carbon Footprint Emission Factors

The analyzer uses internationally recognized emission factors:

| Source | Factor | Unit |
|--------|--------|------|
| Electricity | 0.42 | kg CO₂/kWh |
| Air Travel | 0.255 | kg CO₂/passenger-km |
| Car Travel | 0.21 | kg CO₂/km |
| Train Travel | 0.041 | kg CO₂/passenger-km |
| Servers | 500 | kg CO₂/server/year |
| Cloud Spend | 0.6 | kg CO₂/USD/month |
| Diesel | 2.68 | kg CO₂/liter |
| Gasoline | 2.31 | kg CO₂/liter |
| Office (per employee) | 1,200 | kg CO₂/year |
| Waste | 0.5 | kg CO₂/kg |

## Key Design Principles

- **Modular Code**: Each component (data loading, preprocessing, training, etc.) is in its own module
- **Reproducibility**: Random seeds set for all stochastic operations
- **Documentation**: Comprehensive docstrings and comments
- **Clean ML Pipeline**: Follows best practices for ML project structure
- **Reusability**: Functions designed to be reusable and extensible
- **Testing**: Comprehensive test suite for carbon footprint calculations

## Dataset Information

The project uses the **California Housing dataset**:
- **Samples**: 20,640
- **Features**: 8 numerical features
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude
- **Target**: MedHouseVal (Median house value in $100,000s)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Vivek M

## Acknowledgments

- California Housing dataset from Scikit-learn
- Inspired by real-world regression problems in data science
- Emission factors based on EPA, DEFRA, and IEA standards
