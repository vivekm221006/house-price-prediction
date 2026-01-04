"""
Model Training Module
Trains multiple regression models for house price prediction.
Applies target de-saturation by removing capped house values.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np


# ===============================
# 🔧 HELPER: Remove capped targets
# ===============================
def remove_capped_targets(X, y, cap_value=5.0):
    """
    Remove samples where target value is capped (>= cap_value).

    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target
        cap_value (float): Target cap threshold

    Returns:
        X_filtered, y_filtered
    """
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

    print("Linear Regression training completed!")
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0, random_state=42):
    print("\nTraining Ridge Regression...")
    np.random.seed(random_state)

    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    print(f"Ridge Regression training completed! (alpha={alpha})")
    return model


def train_lasso_regression(X_train, y_train, alpha=1.0, random_state=42):
    print("\nTraining Lasso Regression...")
    np.random.seed(random_state)

    model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)

    print(f"Lasso Regression training completed! (alpha={alpha})")
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

    print(f"Random Forest training completed! (n_estimators={n_estimators})")
    return model


def train_all_models(X_train, y_train, random_state=42):
    """
    Train all regression models AFTER removing capped target values.
    """
    print("=" * 80)
    print("TRAINING ALL MODELS (WITH DE-SATURATION)")
    print("=" * 80)

    # ✂️ REMOVE CAPPED TARGET VALUES (KEY FIX)
    X_train, y_train = remove_capped_targets(X_train, y_train, cap_value=5.0)

    models = {}

    models["Linear Regression"] = train_linear_regression(
        X_train, y_train, random_state
    )

    models["Ridge Regression"] = train_ridge_regression(
        X_train, y_train, alpha=1.0, random_state=random_state
    )

    models["Lasso Regression"] = train_lasso_regression(
        X_train, y_train, alpha=0.1, random_state=random_state
    )

    models["Random Forest"] = train_random_forest(
        X_train, y_train, n_estimators=100, random_state=random_state
    )

    print("\n" + "=" * 80)
    print(f"ALL {len(models)} MODELS TRAINED SUCCESSFULLY (UNCAPPED TARGET)")
    print("=" * 80)

    return models


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
