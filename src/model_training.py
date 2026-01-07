"""
Model Training Module
Trains multiple regression models for house price prediction.
Applies target de-saturation by removing capped house values.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os


# ===============================
# 🔧 HELPER: Remove capped targets
# ===============================
def remove_capped_targets(X, y, cap_value=5.0):
    mask = y < cap_value
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"Removed {len(y) - len(y_filtered)} capped samples (y >= {cap_value})")
    return X_filtered, y_filtered


def train_linear_regression(X_train, y_train, random_state=42):
    print("\nTraining Linear Regression...")
    np.random.seed(random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_ridge_regression(X_train, y_train, alpha=1.0, random_state=42):
    print("\nTraining Ridge Regression...")
    np.random.seed(random_state)

    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    return model


def train_lasso_regression(X_train, y_train, alpha=1.0, random_state=42):
    print("\nTraining Lasso Regression...")
    np.random.seed(random_state)

    model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    print("\nTraining Random Forest Regressor...")
    np.random.seed(random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def train_all_models(X_train, y_train, random_state=42):
    print("=" * 80)
    print("TRAINING ALL MODELS (WITH DE-SATURATION)")
    print("=" * 80)

    # ✂️ REMOVE CAPPED TARGET VALUES
    X_train, y_train = remove_capped_targets(X_train, y_train)

    models = {
        "Linear Regression": train_linear_regression(X_train, y_train, random_state),
        "Ridge Regression": train_ridge_regression(X_train, y_train, 1.0, random_state),
        "Lasso Regression": train_lasso_regression(X_train, y_train, 0.1, random_state),
        "Random Forest": train_random_forest(X_train, y_train, 100, random_state),
    }

    print("ALL MODELS TRAINED SUCCESSFULLY")
    return models


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved → {filepath}")


def load_model(filepath):
    return joblib.load(filepath)


# ===============================
# 🚀 ENTRY POINT (EXECUTION)
# ===============================
if __name__ == "__main__":

    from src.data_loader import load_dataset
    from src.feature_engineering import create_interaction_features
    import os

    print("\n🔁 Running training pipeline...")

    # Ensure directories exist
    os.makedirs("outputs/models", exist_ok=True)

    # Load dataset
    df = load_dataset()

    # Separate features & target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Feature engineering (interaction features)
    X = create_interaction_features(X)

    # Train models (includes de-saturation)
    models = train_all_models(X, y)

    # Save models
    save_model(models["Random Forest"], "outputs/models/random_forest.pkl")
    save_model(models["Linear Regression"], "outputs/models/linear_regression.pkl")
    save_model(models["Ridge Regression"], "outputs/models/ridge_regression.pkl")
    save_model(models["Lasso Regression"], "outputs/models/lasso_regression.pkl")

    print("✅ Training completed successfully")
