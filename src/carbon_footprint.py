"""
Carbon Footprint Analysis Module
AI-based carbon footprint analyzer for organizations.
Calculates emissions from electricity, travel, data centers, fuel, and office operations.
Uses ML models to classify emission levels and generate reduction recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ===============================
# 📊 EMISSION FACTORS (kg CO₂)
# ===============================

EMISSION_FACTORS = {
    "electricity": {
        "kwh": 0.42,  # kg CO₂ per kWh (global average grid factor)
    },
    "business_travel": {
        "air_km": 0.255,     # kg CO₂ per passenger-km (air)
        "car_km": 0.21,      # kg CO₂ per km (average car)
        "train_km": 0.041,   # kg CO₂ per passenger-km (train)
    },
    "data_center": {
        "servers": 500.0,         # kg CO₂ per server per year
        "cloud_spend_usd": 0.6,   # kg CO₂ per $ monthly cloud spend
    },
    "fuel": {
        "diesel_liters": 2.68,    # kg CO₂ per liter diesel
        "gasoline_liters": 2.31,  # kg CO₂ per liter gasoline
    },
    "office": {
        "employees": 1200.0,      # kg CO₂ per employee per year (baseline)
        "waste_kg": 0.5,          # kg CO₂ per kg waste
    },
}

# Emission level thresholds (annual tonnes CO₂)
EMISSION_TIERS = {
    "Low": (0, 50),
    "Moderate": (50, 200),
    "High": (200, 1000),
    "Very High": (1000, float("inf")),
}


# ===============================
# 🔧 EMISSION CALCULATIONS
# ===============================

def calculate_electricity_emissions(monthly_kwh):
    """
    Calculate annual CO₂ emissions from electricity consumption.

    Args:
        monthly_kwh (float): Monthly electricity usage in kWh

    Returns:
        float: Annual CO₂ emissions in kg
    """
    annual_kwh = monthly_kwh * 12
    return annual_kwh * EMISSION_FACTORS["electricity"]["kwh"]


def calculate_travel_emissions(annual_air_km, annual_car_km, annual_train_km):
    """
    Calculate annual CO₂ emissions from business travel.

    Args:
        annual_air_km (float): Annual air travel in passenger-km
        annual_car_km (float): Annual car travel in km
        annual_train_km (float): Annual train travel in passenger-km

    Returns:
        float: Annual CO₂ emissions in kg
    """
    factors = EMISSION_FACTORS["business_travel"]
    air = annual_air_km * factors["air_km"]
    car = annual_car_km * factors["car_km"]
    train = annual_train_km * factors["train_km"]
    return air + car + train


def calculate_data_center_emissions(num_servers, monthly_cloud_spend):
    """
    Calculate annual CO₂ emissions from data centers and cloud usage.

    Args:
        num_servers (int): Number of on-premise servers
        monthly_cloud_spend (float): Monthly cloud service spend in USD

    Returns:
        float: Annual CO₂ emissions in kg
    """
    factors = EMISSION_FACTORS["data_center"]
    servers = num_servers * factors["servers"]
    cloud = monthly_cloud_spend * 12 * factors["cloud_spend_usd"]
    return servers + cloud


def calculate_fuel_emissions(monthly_diesel_liters, monthly_gasoline_liters):
    """
    Calculate annual CO₂ emissions from fuel consumption.

    Args:
        monthly_diesel_liters (float): Monthly diesel consumption in liters
        monthly_gasoline_liters (float): Monthly gasoline consumption in liters

    Returns:
        float: Annual CO₂ emissions in kg
    """
    factors = EMISSION_FACTORS["fuel"]
    diesel = monthly_diesel_liters * 12 * factors["diesel_liters"]
    gasoline = monthly_gasoline_liters * 12 * factors["gasoline_liters"]
    return diesel + gasoline


def calculate_office_emissions(num_employees, monthly_waste_kg):
    """
    Calculate annual CO₂ emissions from office operations.

    Args:
        num_employees (int): Number of employees
        monthly_waste_kg (float): Monthly office waste in kg

    Returns:
        float: Annual CO₂ emissions in kg
    """
    factors = EMISSION_FACTORS["office"]
    employees = num_employees * factors["employees"]
    waste = monthly_waste_kg * 12 * factors["waste_kg"]
    return employees + waste


def calculate_total_emissions(inputs):
    """
    Calculate total annual CO₂ emissions across all categories.

    Args:
        inputs (dict): Dictionary with keys:
            - monthly_kwh, annual_air_km, annual_car_km, annual_train_km,
            - num_servers, monthly_cloud_spend,
            - monthly_diesel_liters, monthly_gasoline_liters,
            - num_employees, monthly_waste_kg

    Returns:
        dict: Emissions breakdown by category and total (in kg CO₂)
    """
    electricity = calculate_electricity_emissions(inputs.get("monthly_kwh", 0))
    travel = calculate_travel_emissions(
        inputs.get("annual_air_km", 0),
        inputs.get("annual_car_km", 0),
        inputs.get("annual_train_km", 0),
    )
    data_center = calculate_data_center_emissions(
        inputs.get("num_servers", 0),
        inputs.get("monthly_cloud_spend", 0),
    )
    fuel = calculate_fuel_emissions(
        inputs.get("monthly_diesel_liters", 0),
        inputs.get("monthly_gasoline_liters", 0),
    )
    office = calculate_office_emissions(
        inputs.get("num_employees", 0),
        inputs.get("monthly_waste_kg", 0),
    )

    total = electricity + travel + data_center + fuel + office

    return {
        "Electricity": electricity,
        "Business Travel": travel,
        "Data Centers & Cloud": data_center,
        "Fuel Consumption": fuel,
        "Office Operations": office,
        "Total": total,
    }


# ===============================
# 🤖 AI-BASED ANALYSIS
# ===============================

def _generate_training_data(n_samples=2000, random_state=42):
    """
    Generate synthetic organizational emission data for training the AI model.

    Args:
        n_samples (int): Number of synthetic organizations to generate
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (features DataFrame, emission tier labels, total emissions)
    """
    np.random.seed(random_state)

    data = pd.DataFrame({
        "monthly_kwh": np.random.uniform(500, 500000, n_samples),
        "annual_air_km": np.random.uniform(0, 5000000, n_samples),
        "annual_car_km": np.random.uniform(0, 1000000, n_samples),
        "annual_train_km": np.random.uniform(0, 500000, n_samples),
        "num_servers": np.random.randint(0, 500, n_samples),
        "monthly_cloud_spend": np.random.uniform(0, 100000, n_samples),
        "monthly_diesel_liters": np.random.uniform(0, 50000, n_samples),
        "monthly_gasoline_liters": np.random.uniform(0, 50000, n_samples),
        "num_employees": np.random.randint(5, 5000, n_samples),
        "monthly_waste_kg": np.random.uniform(50, 50000, n_samples),
    })

    # Calculate total emissions for each synthetic org
    totals = []
    for _, row in data.iterrows():
        result = calculate_total_emissions(row.to_dict())
        totals.append(result["Total"])

    total_emissions = np.array(totals)
    total_tonnes = total_emissions / 1000.0

    # Assign tier labels
    labels = []
    for t in total_tonnes:
        for tier_name, (low, high) in EMISSION_TIERS.items():
            if low <= t < high:
                labels.append(tier_name)
                break

    return data, labels, total_tonnes


def build_emission_classifier(random_state=42):
    """
    Build and train an AI model to classify organizations by emission tier.

    Args:
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (trained classifier, scaler, feature columns)
    """
    data, labels, _ = _generate_training_data(random_state=random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_scaled, labels)

    return clf, scaler, list(data.columns)


def build_reduction_predictor(random_state=42):
    """
    Build an AI model to predict potential emission reduction percentage.
    Models the relationship between current practices and achievable reduction.

    Args:
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (trained regressor, scaler)
    """
    data, _, total_tonnes = _generate_training_data(random_state=random_state)

    # Simulate potential reduction based on profile
    # Higher emitters can typically achieve larger reductions
    np.random.seed(random_state)
    base_reduction = np.clip(10 + 0.01 * total_tonnes + np.random.normal(0, 5, len(total_tonnes)), 5, 45)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    reg = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    reg.fit(X_scaled, base_reduction)

    return reg, scaler


def classify_emissions(inputs, clf=None, scaler=None, feature_cols=None):
    """
    Classify an organization's emission tier using the AI model.

    Args:
        inputs (dict): Organization input data
        clf: Pre-trained classifier (built if None)
        scaler: Pre-fitted scaler (built if None)
        feature_cols: Feature column names

    Returns:
        str: Predicted emission tier
    """
    if clf is None or scaler is None or feature_cols is None:
        clf, scaler, feature_cols = build_emission_classifier()

    input_df = pd.DataFrame([{col: inputs.get(col, 0) for col in feature_cols}])
    X_scaled = scaler.transform(input_df)
    prediction = clf.predict(X_scaled)[0]
    return prediction


def predict_reduction_potential(inputs, reg=None, scaler=None):
    """
    Predict achievable emission reduction percentage for an organization.

    Args:
        inputs (dict): Organization input data
        reg: Pre-trained regressor (built if None)
        scaler: Pre-fitted scaler (built if None)

    Returns:
        float: Predicted reduction percentage (5-45%)
    """
    if reg is None or scaler is None:
        reg, scaler = build_reduction_predictor()

    feature_cols = [
        "monthly_kwh", "annual_air_km", "annual_car_km", "annual_train_km",
        "num_servers", "monthly_cloud_spend",
        "monthly_diesel_liters", "monthly_gasoline_liters",
        "num_employees", "monthly_waste_kg",
    ]
    input_df = pd.DataFrame([{col: inputs.get(col, 0) for col in feature_cols}])
    X_scaled = scaler.transform(input_df)
    reduction = reg.predict(X_scaled)[0]
    return float(np.clip(reduction, 5, 45))


# ===============================
# 💡 RECOMMENDATION ENGINE
# ===============================

RECOMMENDATIONS = {
    "Electricity": [
        "Switch to renewable energy providers or install on-site solar panels",
        "Upgrade to energy-efficient LED lighting and HVAC systems",
        "Implement smart building energy management systems",
        "Schedule energy-intensive operations during off-peak hours",
    ],
    "Business Travel": [
        "Replace short-haul flights with virtual meetings or train travel",
        "Implement a company-wide travel policy prioritizing low-carbon options",
        "Invest in electric or hybrid company fleet vehicles",
        "Encourage carpooling and public transit with subsidized passes",
    ],
    "Data Centers & Cloud": [
        "Migrate workloads to cloud providers powered by renewable energy",
        "Optimize server utilization and decommission underused hardware",
        "Implement virtualization and containerization to reduce server count",
        "Use auto-scaling to avoid over-provisioning cloud resources",
    ],
    "Fuel Consumption": [
        "Transition fleet vehicles to electric or hybrid alternatives",
        "Optimize logistics and delivery routes to reduce fuel usage",
        "Implement anti-idling policies for fleet vehicles",
        "Consider biofuels or alternative fuels for heavy-duty vehicles",
    ],
    "Office Operations": [
        "Implement comprehensive recycling and composting programs",
        "Adopt remote or hybrid work policies to reduce office footprint",
        "Use sustainable procurement for office supplies and furniture",
        "Install motion-sensor lighting and smart thermostats",
    ],
}


def generate_recommendations(emissions_breakdown):
    """
    Generate AI-driven recommendations based on emission profile.
    Prioritizes categories with the highest emissions.

    Args:
        emissions_breakdown (dict): Emissions by category from calculate_total_emissions()

    Returns:
        list: Prioritized list of (category, recommendation) tuples
    """
    # Exclude 'Total' and sort categories by emission level (descending)
    categories = {
        k: v for k, v in emissions_breakdown.items() if k != "Total"
    }
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    recommendations = []
    for category, emission_kg in sorted_categories:
        if emission_kg > 0 and category in RECOMMENDATIONS:
            # Pick top 2 recommendations for each category
            recs = RECOMMENDATIONS[category][:2]
            for rec in recs:
                recommendations.append((category, rec))

    return recommendations


def get_emission_tier(total_tonnes):
    """
    Determine the emission tier based on total annual tonnes CO₂.

    Args:
        total_tonnes (float): Total annual emissions in tonnes CO₂

    Returns:
        str: Emission tier name
    """
    for tier_name, (low, high) in EMISSION_TIERS.items():
        if low <= total_tonnes < high:
            return tier_name
    return "Very High"
