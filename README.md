House Price Prediction

An end-to-end machine learning project that predicts house prices using the California Housing Dataset.
The project demonstrates a complete ML pipeline including data preprocessing, feature engineering, model training, evaluation, and visualization, along with an interactive Streamlit web application.

🔗 Live Website:
https://house-price-prediction-v9tf6xoigyg2y2qfxekkwh.streamlit.app/

Project Overview

This project demonstrates a full machine learning workflow:

Data loading and exploration

Data preprocessing and cleaning

Feature engineering

Training multiple regression models

Model evaluation using multiple metrics

Data visualization

Deployment using Streamlit

The goal is to predict median house values based on housing and geographical features.

Features
🏠 House Price Prediction
Data Loading

Loads the California Housing Dataset provided by Scikit-learn.

Data Preprocessing

Handle missing values

Encode categorical features

Scale numerical features

Feature Engineering

Create interaction features

Feature selection using univariate methods

Feature importance analysis

Model Training

Multiple regression models are trained and compared:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Model Evaluation

Models are evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Visualizations

The project generates several visualizations:

Correlation heatmap

Feature distributions

Feature vs price relationships

Feature importance

Actual vs predicted prices

Residual plots

Model comparison charts

Project Structure
house-price-prediction/
│
├── data/                          # Dataset directory
│   └── housing_data_raw.csv
│
├── outputs/
│   ├── models/                    # Saved trained models
│   │   ├── linear_regression.pkl
│   │   ├── ridge_regression.pkl
│   │   ├── lasso_regression.pkl
│   │   └── random_forest.pkl
│   │
│   ├── plots/                     # Generated visualizations
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions.png
│   │   ├── feature_vs_target.png
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals.png
│   │   └── model_comparison.png
│   │
│   └── model_metrics.csv
│
├── src/                           # Source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── visualization.py
│
├── app.py                         # Main ML pipeline
├── streamlit_app.py               # Streamlit web application
├── requirements.txt               # Dependencies
└── README.md
Tech Stack
Programming Language

Python 3.7+

Libraries

NumPy – Numerical computing

Pandas – Data manipulation and analysis

Scikit-learn – Machine learning models

Matplotlib – Data visualization

Seaborn – Statistical plots

Streamlit – Interactive web application

Installation
1. Clone the Repository
git clone https://github.com/vivekm221006/house-price-prediction.git
cd house-price-prediction
2. Install Dependencies
pip install -r requirements.txt
Usage
Run the ML Pipeline
python app.py

This will:

Load the California Housing dataset

Perform data preprocessing and cleaning

Create engineered features

Train multiple regression models

Evaluate model performance

Generate visualizations

Save outputs to the outputs/ directory

Run the Streamlit Web Application
streamlit run streamlit_app.py

This launches an interactive dashboard where users can enter housing parameters and get real-time price predictions.

Outputs

After running the pipeline you will get:

Dataset
data/housing_data_raw.csv
Trained Models
outputs/models/*.pkl
Evaluation Metrics
outputs/model_metrics.csv
Visualizations
outputs/plots/*.png
Model Performance

The pipeline compares four regression models:

Model	Description
Linear Regression	Baseline linear regression model
Ridge Regression	Linear regression with L2 regularization
Lasso Regression	Linear regression with L1 regularization
Random Forest	Ensemble tree-based regression model

Each model is evaluated using MAE, MSE, RMSE, and R² Score on training and testing datasets.

Dataset Information

The project uses the California Housing Dataset.

Total Samples: 20,640
Features: 8 numerical features

Feature	Description
MedInc	Median income in block group
HouseAge	Median house age
AveRooms	Average number of rooms
AveBedrms	Average bedrooms
Population	Block group population
AveOccup	Average household members
Latitude	Block latitude
Longitude	Block longitude

Target Variable

MedHouseVal

Median house value in $100,000 units.

License

This project is licensed under the MIT License.

Author

Vivek M

Acknowledgments

California Housing dataset from Scikit-learn

Inspired by real-world housing price prediction problems in data science
