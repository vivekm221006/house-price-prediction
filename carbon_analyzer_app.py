"""
Carbon Footprint Analyzer - Streamlit Application
AI-based carbon footprint analyzer for organizations.
Analyzes emissions and recommends ways to reduce them.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.carbon_footprint import (
    calculate_total_emissions,
    classify_emissions,
    predict_reduction_potential,
    generate_recommendations,
    build_emission_classifier,
    build_reduction_predictor,
    get_emission_tier,
    EMISSION_FACTORS,
)

# Page config
st.set_page_config(
    page_title="Carbon Footprint Analyzer",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 AI-Based Carbon Footprint Analyzer")
st.markdown("### Measure, Analyze & Reduce Your Organization's Carbon Emissions")
st.markdown(
    "Enter your organization's data below to get an AI-powered carbon "
    "footprint analysis with actionable reduction recommendations."
)

st.markdown("---")

# ===============================
# 📝 INPUT FORM
# ===============================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚡ Electricity Consumption")
    monthly_kwh = st.number_input(
        "Monthly Electricity Usage (kWh)",
        min_value=0.0, max_value=10000000.0, value=50000.0, step=1000.0,
        help="Total monthly electricity consumption in kilowatt-hours",
    )

    st.subheader("✈️ Business Travel (Annual)")
    annual_air_km = st.number_input(
        "Air Travel (passenger-km/year)",
        min_value=0.0, max_value=50000000.0, value=500000.0, step=10000.0,
        help="Total annual air travel distance in passenger-kilometers",
    )
    annual_car_km = st.number_input(
        "Car Travel (km/year)",
        min_value=0.0, max_value=10000000.0, value=200000.0, step=5000.0,
        help="Total annual car travel distance in kilometers",
    )
    annual_train_km = st.number_input(
        "Train Travel (passenger-km/year)",
        min_value=0.0, max_value=10000000.0, value=100000.0, step=5000.0,
        help="Total annual train travel distance in passenger-kilometers",
    )

    st.subheader("🏢 Office Operations")
    num_employees = st.number_input(
        "Number of Employees",
        min_value=1, max_value=100000, value=100, step=10,
        help="Total number of employees in the organization",
    )
    monthly_waste_kg = st.number_input(
        "Monthly Office Waste (kg)",
        min_value=0.0, max_value=1000000.0, value=2000.0, step=100.0,
        help="Total monthly office waste in kilograms",
    )

with col_right:
    st.subheader("🖥️ Data Centers & Cloud")
    num_servers = st.number_input(
        "Number of On-Premise Servers",
        min_value=0, max_value=10000, value=20, step=1,
        help="Number of physical servers owned by the organization",
    )
    monthly_cloud_spend = st.number_input(
        "Monthly Cloud Spend (USD)",
        min_value=0.0, max_value=10000000.0, value=5000.0, step=500.0,
        help="Monthly cloud service expenditure in US dollars",
    )

    st.subheader("⛽ Fuel Consumption (Monthly)")
    monthly_diesel_liters = st.number_input(
        "Diesel Consumption (liters/month)",
        min_value=0.0, max_value=1000000.0, value=5000.0, step=100.0,
        help="Monthly diesel fuel consumption in liters",
    )
    monthly_gasoline_liters = st.number_input(
        "Gasoline Consumption (liters/month)",
        min_value=0.0, max_value=1000000.0, value=3000.0, step=100.0,
        help="Monthly gasoline fuel consumption in liters",
    )

st.markdown("---")

# ===============================
# 🔮 ANALYSIS
# ===============================

if st.button("🔮 Analyze Carbon Footprint", use_container_width=True, type="primary"):

    inputs = {
        "monthly_kwh": monthly_kwh,
        "annual_air_km": annual_air_km,
        "annual_car_km": annual_car_km,
        "annual_train_km": annual_train_km,
        "num_servers": num_servers,
        "monthly_cloud_spend": monthly_cloud_spend,
        "monthly_diesel_liters": monthly_diesel_liters,
        "monthly_gasoline_liters": monthly_gasoline_liters,
        "num_employees": num_employees,
        "monthly_waste_kg": monthly_waste_kg,
    }

    with st.spinner("🤖 Running AI analysis..."):

        # Calculate emissions
        emissions = calculate_total_emissions(inputs)
        total_kg = emissions["Total"]
        total_tonnes = total_kg / 1000.0

        # Build AI models and classify
        clf, clf_scaler, feature_cols = build_emission_classifier()
        ai_tier = classify_emissions(inputs, clf, clf_scaler, feature_cols)

        reg, reg_scaler = build_reduction_predictor()
        reduction_pct = predict_reduction_potential(inputs, reg, reg_scaler)

        # Rule-based tier for comparison
        rule_tier = get_emission_tier(total_tonnes)

    # =========================================================================
    # RESULTS DISPLAY
    # =========================================================================

    st.success("✅ Analysis Complete")

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Emissions", f"{total_tonnes:,.1f} tonnes CO₂/year")
    m2.metric("AI Emission Tier", ai_tier)
    m3.metric("Reduction Potential", f"{reduction_pct:.1f}%")
    m4.metric(
        "Potential Savings",
        f"{total_tonnes * reduction_pct / 100:,.1f} tonnes CO₂/year",
    )

    st.markdown("---")

    # ----- Breakdown Table -----
    chart_col, table_col = st.columns(2)

    with table_col:
        st.subheader("📋 Emissions Breakdown")
        breakdown_data = {
            k: v for k, v in emissions.items() if k != "Total"
        }
        df_breakdown = pd.DataFrame({
            "Category": breakdown_data.keys(),
            "Annual Emissions (kg CO₂)": [f"{v:,.0f}" for v in breakdown_data.values()],
            "Annual Emissions (tonnes CO₂)": [f"{v / 1000:,.1f}" for v in breakdown_data.values()],
            "Share (%)": [f"{v / total_kg * 100:.1f}" if total_kg > 0 else "0.0" for v in breakdown_data.values()],
        })
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    # ----- Pie Chart -----
    with chart_col:
        st.subheader("📊 Emission Distribution")
        categories = list(breakdown_data.keys())
        values = list(breakdown_data.values())

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ["#f39c12", "#3498db", "#9b59b6", "#e74c3c", "#2ecc71"]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=categories,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            textprops={"fontsize": 9},
        )
        ax.set_title("Carbon Emissions by Category", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ----- Bar Chart -----
    st.subheader("📈 Emissions by Category (tonnes CO₂/year)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    tonnes_values = [v / 1000 for v in values]
    bars = ax2.barh(categories, tonnes_values, color=colors)
    ax2.set_xlabel("Annual Emissions (tonnes CO₂)")
    ax2.set_title("Carbon Footprint by Category")
    for bar, val in zip(bars, tonnes_values):
        ax2.text(
            bar.get_width() + max(tonnes_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.1f}",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    # ----- AI Recommendations -----
    st.subheader("💡 AI-Powered Reduction Recommendations")
    st.markdown(
        f"Based on your emission profile (**{ai_tier}** tier), "
        f"our AI estimates you can reduce emissions by **{reduction_pct:.1f}%** "
        f"(~**{total_tonnes * reduction_pct / 100:,.1f} tonnes CO₂/year**)."
    )

    recommendations = generate_recommendations(emissions)

    current_category = None
    for category, rec in recommendations:
        if category != current_category:
            st.markdown(f"**{category}:**")
            current_category = category
        st.markdown(f"- ✅ {rec}")

    st.markdown("---")

    # ----- Emission Factors Reference -----
    with st.expander("📖 Emission Factors Used"):
        st.markdown("**Electricity:** 0.42 kg CO₂/kWh (global average)")
        st.markdown("**Air Travel:** 0.255 kg CO₂/passenger-km")
        st.markdown("**Car Travel:** 0.21 kg CO₂/km")
        st.markdown("**Train Travel:** 0.041 kg CO₂/passenger-km")
        st.markdown("**Servers:** 500 kg CO₂/server/year")
        st.markdown("**Cloud Spend:** 0.6 kg CO₂/USD/month")
        st.markdown("**Diesel:** 2.68 kg CO₂/liter")
        st.markdown("**Gasoline:** 2.31 kg CO₂/liter")
        st.markdown("**Office (per employee):** 1,200 kg CO₂/year")
        st.markdown("**Waste:** 0.5 kg CO₂/kg")

# Footer
st.markdown("---")
st.caption(
    "🌍 AI-Based Carbon Footprint Analyzer | "
    "Emission factors based on international standards (EPA, DEFRA, IEA). "
    "Results are estimates for planning purposes."
)
