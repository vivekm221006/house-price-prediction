"""
Tests for the Carbon Footprint Analysis Module.
"""

import pytest
import numpy as np
from src.carbon_footprint import (
    calculate_electricity_emissions,
    calculate_travel_emissions,
    calculate_data_center_emissions,
    calculate_fuel_emissions,
    calculate_office_emissions,
    calculate_total_emissions,
    generate_recommendations,
    get_emission_tier,
    classify_emissions,
    predict_reduction_potential,
    build_emission_classifier,
    build_reduction_predictor,
    EMISSION_FACTORS,
    EMISSION_TIERS,
)


@pytest.fixture
def sample_org_inputs():
    """Sample organization inputs used across multiple tests."""
    return {
        "monthly_kwh": 50000, "annual_air_km": 500000,
        "annual_car_km": 200000, "annual_train_km": 100000,
        "num_servers": 20, "monthly_cloud_spend": 5000,
        "monthly_diesel_liters": 5000, "monthly_gasoline_liters": 3000,
        "num_employees": 100, "monthly_waste_kg": 2000,
    }


# ===============================
# Emission Calculation Tests
# ===============================

class TestElectricityEmissions:
    def test_zero_usage(self):
        assert calculate_electricity_emissions(0) == 0.0

    def test_known_value(self):
        # 10,000 kWh/month * 12 months * 0.42 factor = 50,400 kg
        result = calculate_electricity_emissions(10000)
        assert result == pytest.approx(50400.0)

    def test_positive_usage(self):
        result = calculate_electricity_emissions(5000)
        assert result > 0


class TestTravelEmissions:
    def test_zero_travel(self):
        assert calculate_travel_emissions(0, 0, 0) == 0.0

    def test_air_only(self):
        result = calculate_travel_emissions(100000, 0, 0)
        expected = 100000 * EMISSION_FACTORS["business_travel"]["air_km"]
        assert result == pytest.approx(expected)

    def test_car_only(self):
        result = calculate_travel_emissions(0, 50000, 0)
        expected = 50000 * EMISSION_FACTORS["business_travel"]["car_km"]
        assert result == pytest.approx(expected)

    def test_train_only(self):
        result = calculate_travel_emissions(0, 0, 30000)
        expected = 30000 * EMISSION_FACTORS["business_travel"]["train_km"]
        assert result == pytest.approx(expected)

    def test_combined_travel(self):
        result = calculate_travel_emissions(100000, 50000, 30000)
        assert result > 0
        # Air should contribute the most
        air_only = calculate_travel_emissions(100000, 0, 0)
        assert air_only < result


class TestDataCenterEmissions:
    def test_zero_inputs(self):
        assert calculate_data_center_emissions(0, 0) == 0.0

    def test_servers_only(self):
        result = calculate_data_center_emissions(10, 0)
        expected = 10 * EMISSION_FACTORS["data_center"]["servers"]
        assert result == pytest.approx(expected)

    def test_cloud_only(self):
        result = calculate_data_center_emissions(0, 5000)
        expected = 5000 * 12 * EMISSION_FACTORS["data_center"]["cloud_spend_usd"]
        assert result == pytest.approx(expected)


class TestFuelEmissions:
    def test_zero_fuel(self):
        assert calculate_fuel_emissions(0, 0) == 0.0

    def test_diesel_only(self):
        result = calculate_fuel_emissions(1000, 0)
        expected = 1000 * 12 * EMISSION_FACTORS["fuel"]["diesel_liters"]
        assert result == pytest.approx(expected)

    def test_gasoline_only(self):
        result = calculate_fuel_emissions(0, 1000)
        expected = 1000 * 12 * EMISSION_FACTORS["fuel"]["gasoline_liters"]
        assert result == pytest.approx(expected)

    def test_diesel_higher_than_gasoline(self):
        diesel = calculate_fuel_emissions(1000, 0)
        gasoline = calculate_fuel_emissions(0, 1000)
        assert diesel > gasoline  # Diesel has higher emission factor


class TestOfficeEmissions:
    def test_zero_inputs(self):
        assert calculate_office_emissions(0, 0) == 0.0

    def test_employees_only(self):
        result = calculate_office_emissions(100, 0)
        expected = 100 * EMISSION_FACTORS["office"]["employees"]
        assert result == pytest.approx(expected)

    def test_waste_only(self):
        result = calculate_office_emissions(0, 500)
        expected = 500 * 12 * EMISSION_FACTORS["office"]["waste_kg"]
        assert result == pytest.approx(expected)


class TestTotalEmissions:
    def test_all_zeros(self):
        inputs = {
            "monthly_kwh": 0, "annual_air_km": 0, "annual_car_km": 0,
            "annual_train_km": 0, "num_servers": 0, "monthly_cloud_spend": 0,
            "monthly_diesel_liters": 0, "monthly_gasoline_liters": 0,
            "num_employees": 0, "monthly_waste_kg": 0,
        }
        result = calculate_total_emissions(inputs)
        assert result["Total"] == 0.0
        for key in result:
            assert result[key] == 0.0

    def test_returns_all_categories(self):
        inputs = {"monthly_kwh": 1000, "num_employees": 10}
        result = calculate_total_emissions(inputs)
        assert "Electricity" in result
        assert "Business Travel" in result
        assert "Data Centers & Cloud" in result
        assert "Fuel Consumption" in result
        assert "Office Operations" in result
        assert "Total" in result

    def test_total_is_sum_of_categories(self, sample_org_inputs):
        result = calculate_total_emissions(sample_org_inputs)
        category_sum = sum(v for k, v in result.items() if k != "Total")
        assert result["Total"] == pytest.approx(category_sum)

    def test_missing_keys_default_to_zero(self):
        inputs = {"monthly_kwh": 10000}
        result = calculate_total_emissions(inputs)
        assert result["Total"] > 0
        assert result["Electricity"] > 0
        assert result["Business Travel"] == 0.0
        assert result["Data Centers & Cloud"] == 0.0
        assert result["Fuel Consumption"] == 0.0
        assert result["Office Operations"] == 0.0


# ===============================
# Emission Tier Tests
# ===============================

class TestEmissionTier:
    def test_low_tier(self):
        assert get_emission_tier(10) == "Low"

    def test_moderate_tier(self):
        assert get_emission_tier(100) == "Moderate"

    def test_high_tier(self):
        assert get_emission_tier(500) == "High"

    def test_very_high_tier(self):
        assert get_emission_tier(5000) == "Very High"

    def test_boundary_low_moderate(self):
        assert get_emission_tier(50) == "Moderate"

    def test_boundary_moderate_high(self):
        assert get_emission_tier(200) == "High"

    def test_boundary_high_very_high(self):
        assert get_emission_tier(1000) == "Very High"

    def test_zero_emissions(self):
        assert get_emission_tier(0) == "Low"


# ===============================
# Recommendation Tests
# ===============================

class TestRecommendations:
    def test_returns_recommendations(self):
        emissions = {
            "Electricity": 50000,
            "Business Travel": 30000,
            "Data Centers & Cloud": 10000,
            "Fuel Consumption": 20000,
            "Office Operations": 5000,
            "Total": 115000,
        }
        recs = generate_recommendations(emissions)
        assert len(recs) > 0

    def test_highest_emitter_first(self):
        emissions = {
            "Electricity": 100000,
            "Business Travel": 10000,
            "Data Centers & Cloud": 5000,
            "Fuel Consumption": 1000,
            "Office Operations": 500,
            "Total": 116500,
        }
        recs = generate_recommendations(emissions)
        assert recs[0][0] == "Electricity"

    def test_zero_emissions_no_recs(self):
        emissions = {
            "Electricity": 0,
            "Business Travel": 0,
            "Data Centers & Cloud": 0,
            "Fuel Consumption": 0,
            "Office Operations": 0,
            "Total": 0,
        }
        recs = generate_recommendations(emissions)
        assert len(recs) == 0

    def test_each_recommendation_is_tuple(self):
        emissions = {
            "Electricity": 50000,
            "Business Travel": 30000,
            "Data Centers & Cloud": 0,
            "Fuel Consumption": 0,
            "Office Operations": 0,
            "Total": 80000,
        }
        recs = generate_recommendations(emissions)
        for rec in recs:
            assert isinstance(rec, tuple)
            assert len(rec) == 2


# ===============================
# AI Model Tests
# ===============================

class TestAIModels:
    def test_classifier_builds(self):
        clf, scaler, feature_cols = build_emission_classifier()
        assert clf is not None
        assert scaler is not None
        assert len(feature_cols) == 10

    def test_classifier_predicts_valid_tier(self, sample_org_inputs):
        clf, scaler, feature_cols = build_emission_classifier()
        tier = classify_emissions(sample_org_inputs, clf, scaler, feature_cols)
        assert tier in EMISSION_TIERS

    def test_reduction_predictor_builds(self):
        reg, scaler = build_reduction_predictor()
        assert reg is not None
        assert scaler is not None

    def test_reduction_in_valid_range(self, sample_org_inputs):
        reg, scaler = build_reduction_predictor()
        reduction = predict_reduction_potential(sample_org_inputs, reg, scaler)
        assert 5.0 <= reduction <= 45.0
