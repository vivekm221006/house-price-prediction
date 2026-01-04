import shap
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame
df = df[df["MedHouseVal"] < 5.0]

X = df.drop(columns=["MedHouseVal"])

X["RoomsPerBedroom"] = X["AveRooms"] / (X["AveBedrms"] + 0.01)
X["RoomsPerPerson"] = X["AveRooms"] / (X["AveOccup"] + 0.01)
X["HouseholdsPerPopulation"] = X["AveOccup"] / (X["Population"] + 0.01)

model = joblib.load("outputs/models/random_forest.pkl")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.sample(1000, random_state=42))

shap.summary_plot(shap_values, X.sample(1000, random_state=42))
