# 🏠 House Price Prediction

An end-to-end Machine Learning project that predicts house prices using the **California Housing Dataset**.  
The project demonstrates a complete ML pipeline including **data preprocessing, feature engineering, model training, evaluation, visualization, and deployment using Streamlit**.

🔗 **Live Website:**  
https://house-price-prediction-v9tf6xoigyg2y2qfxekkwh.streamlit.app/

---

# 📌 Project Overview

This project demonstrates a full **Machine Learning workflow**:

- Data loading and exploration
- Data preprocessing and cleaning
- Feature engineering
- Training multiple regression models
- Model evaluation using multiple metrics
- Data visualization
- Deployment using **Streamlit Web App**

The goal is to predict **median house prices** based on housing features and geographic information.

---

# 🚀 Features

## 🏡 House Price Prediction

### Data Loading
Loads the **California Housing Dataset** provided by Scikit-learn.

### Data Preprocessing
- Handle missing values
- Encode categorical features
- Scale numerical features

### Feature Engineering
- Create interaction features
- Feature selection using univariate methods
- Feature importance analysis

### Model Training
Multiple regression models are trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

### Model Evaluation
Models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Data Visualizations

The project generates several visualizations:

- Correlation heatmap
- Feature distributions
- Feature vs price relationships
- Feature importance
- Actual vs predicted prices
- Residual plots
- Model comparison charts

---

# 📂 Project Structure
house-price-prediction/
│
├── data/
│ └── housing_data_raw.csv
│
├── outputs/
│ ├── models/
│ │ ├── linear_regression.pkl
│ │ ├── ridge_regression.pkl
│ │ ├── lasso_regression.pkl
│ │ └── random_forest.pkl
│ │
│ ├── plots/
│ │ ├── correlation_heatmap.png
│ │ ├── feature_distributions.png
│ │ ├── feature_vs_target.png
│ │ ├── feature_importance.png
│ │ ├── actual_vs_predicted.png
│ │ ├── residuals.png
│ │ └── model_comparison.png
│ │
│ └── model_metrics.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── visualization.py
│
├── app.py
├── streamlit_app.py
├── requirements.txt
└── README.md



---

# 🛠 Tech Stack

### Programming Language
- Python 3.7+

### Libraries Used

- **NumPy** – Numerical computing  
- **Pandas** – Data analysis and manipulation  
- **Scikit-learn** – Machine learning models  
- **Matplotlib** – Data visualization  
- **Seaborn** – Statistical visualization  
- **Streamlit** – Interactive web application  

---

# ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vivekm221006/house-price-prediction.git
cd house-price-prediction

2️⃣ Install Dependencies

pip install -r requirements.txt

▶️ Usage
Run the Machine Learning Pipeline
python app.py

This will:
1) Load the California Housing dataset
2) Perform data preprocessing
3) Generate engineered features
4) Train multiple regression models
5) Evaluate model performance
6) Generate visualizations
7) Save outputs to the outputs/ directory

Run the Streamlit Web Application
streamlit run streamlit_app.py


This launches an interactive web application where users can input housing features and get real-time house price predictions.

📊 Outputs:

After running the pipeline you will get:

Dataset
data/housing_data_raw.csv
Trained Models
outputs/models/*.pkl
Evaluation Metrics
outputs/model_metrics.csv
Visualizations
outputs/plots/*.png
📈 Model Performance

The pipeline compares four regression models:

Model	Description
Linear Regression	Baseline linear regression model
Ridge Regression	Linear regression with L2 regularization
Lasso Regression	Linear regression with L1 regularization
Random Forest	Ensemble tree-based regression model

Each model is evaluated using MAE, MSE, RMSE, and R² Score.

📊 Dataset Information

The project uses the California Housing Dataset.

Total Samples: 20,640
Total Features: 8

Feature	Description
MedInc	Median income in block group
HouseAge	Median house age
AveRooms	Average number of rooms
AveBedrms	Average number of bedrooms
Population	Block group population
AveOccup	Average household members
Latitude	Block latitude
Longitude	Block longitude


Target Variable
MedHouseVal

Median house value in $100,000 units.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Vivek M

🙏 Acknowledgments

California Housing Dataset from Scikit-learn

Inspired by real-world housing price prediction problems in data science
