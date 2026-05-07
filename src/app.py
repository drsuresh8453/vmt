"""
app.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Streamlit web application for insurance agents.
Form collects only the 18 features the model was trained on.

18 model features:
    daily_trips, is_rideshare, uses_for_business, annual_commute_km,
    family_trip_km, selling_price, fuel_type, is_high_mileage,
    brand_tier, occupation, is_premium_brand, driving_exp_years,
    highway_access, num_vehicles_owned, home_to_office_km,
    night_driving, has_metro_rail, num_children
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict, get_model_info

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Mileage Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: bold;
        color: #1F4E79; text-align: center; padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚗 Vehicle Annual Mileage Predictor</div>',
            unsafe_allow_html=True)
st.markdown("**DV Analytics | Author: Suresh D R | AI Product Developer & Technology Mentor**")
st.markdown("---")

# ── Sidebar — Model Info ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    try:
        info = get_model_info()
        st.success(f"Model: **{info.get('model', 'Loaded')}**")
        if info.get('mape'):    st.metric("MAPE",     f"{info.get('mape')}%")
        if info.get('mae'):     st.metric("MAE",      f"{info.get('mae'):,} km")
        if info.get('r2'):      st.metric("R² Score", f"{info.get('r2')}")
        if info.get('trained_on'): st.caption(f"Trained: {info.get('trained_on')}")
    except Exception as e:
        st.warning(f"Model info unavailable: {e}")

    st.markdown("---")
    st.markdown("""
    ### Risk Categories
    - 🟢 **Low** — < 10,000 km
    - 🟡 **Medium** — 10,000–25,000 km
    - 🟠 **High** — 25,000–40,000 km
    - 🔴 **Very High** — 40,000–70,000 km
    - 🟣 **Extreme** — > 70,000 km
    """)

# ── Input Form ─────────────────────────────────────────────────────────────
st.header("📝 Vehicle & Owner Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚗 Vehicle Details")

    fuel_type = st.selectbox("Fuel Type",
        ['Petrol', 'Diesel', 'CNG', 'Electric'])

    brand_tier = st.selectbox("Brand Tier",
        ['Budget', 'Mid', 'Premium'])

    is_premium_brand = 1 if brand_tier == 'Premium' else 0
    st.caption(f"Is Premium Brand: {'Yes' if is_premium_brand else 'No'} (auto-set from Brand Tier)")

    selling_price = st.number_input("Selling Price (₹)",
        min_value=100000, max_value=5000000, value=600000, step=50000)

    num_vehicles_owned = st.selectbox("Total Vehicles Owned", [1, 2, 3, 4])

with col2:
    st.subheader("👤 Owner Details")

    occupation = st.selectbox("Occupation",
        ['Salaried', 'Business', 'Student', 'Retired', 'Self-Employed'])

    driving_exp_years = st.slider("Driving Experience (Years)", 1, 40, 8)

    num_children = st.selectbox("Number of Children", [0, 1, 2, 3, 4])

with col3:
    st.subheader("📍 Location & Usage")

    home_to_office_km = st.slider("Home to Office Distance (km)", 0, 80, 18)

    has_metro_rail = st.selectbox("Metro Rail Available in City",
        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    highway_access = st.selectbox("Near Highway",
        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    daily_trips = st.slider("Daily Trips (one-way)", 1, 20, 4)

    st.markdown("**Usage Pattern**")

    is_rideshare = st.selectbox("Rideshare Driver (Ola/Uber)",
        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    uses_for_business = st.selectbox("Uses Vehicle for Business",
        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    night_driving = st.selectbox("Regular Night Driving",
        [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("**Trip Distances**")
    family_trip_km = st.number_input("Family/Weekend Trip Distance (km/month)",
        min_value=0, max_value=5000, value=200, step=50)

# ── Derived Features (computed automatically) ──────────────────────────────
# annual_commute_km = home_to_office * 2 (both ways) * working days
annual_commute_km = home_to_office_km * 2 * 250

# is_high_mileage = 1 if rideshare OR business use OR daily trips > 8
is_high_mileage = 1 if (is_rideshare == 1 or uses_for_business == 1 or daily_trips > 8) else 0

with st.expander("ℹ️ See auto-computed features"):
    st.write(f"**annual_commute_km** = {home_to_office_km} km × 2 × 250 days = **{annual_commute_km:,} km**")
    st.write(f"**is_high_mileage** = **{is_high_mileage}** (1 if rideshare/business/daily trips > 8)")
    st.write(f"**is_premium_brand** = **{is_premium_brand}** (1 if brand tier = Premium)")

# ── Predict Button ─────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 3])
with predict_col:
    predict_btn = st.button("🔮 Predict Annual Mileage", type="primary",
                             use_container_width=True)

# ── Show Prediction ────────────────────────────────────────────────────────
if predict_btn:
    # Build exactly the 18 features the model expects
    record = {
        'daily_trips'       : daily_trips,
        'is_rideshare'      : is_rideshare,
        'uses_for_business' : uses_for_business,
        'annual_commute_km' : annual_commute_km,
        'family_trip_km'    : family_trip_km,
        'selling_price'     : selling_price,
        'fuel_type'         : fuel_type,
        'is_high_mileage'   : is_high_mileage,
        'brand_tier'        : brand_tier,
        'occupation'        : occupation,
        'is_premium_brand'  : is_premium_brand,
        'driving_exp_years' : driving_exp_years,
        'highway_access'    : highway_access,
        'num_vehicles_owned': num_vehicles_owned,
        'home_to_office_km' : home_to_office_km,
        'night_driving'     : night_driving,
        'has_metro_rail'    : has_metro_rail,
        'num_children'      : num_children,
    }

    with st.spinner("Predicting..."):
        try:
            result = predict(record)

            st.markdown("---")
            st.header("📊 Prediction Results")

            # Main metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Annual KMs",
                          f"{result['predicted_kms']:,} km")
            with m2:
                st.metric("Expected Range",
                          f"{result['predicted_kms_range']['low']:,} – "
                          f"{result['predicted_kms_range']['high']:,} km")
            with m3:
                risk_icons = {
                    'Low': '🟢', 'Medium': '🟡',
                    'High': '🟠', 'Very High': '🔴', 'Extreme': '🟣'
                }
                icon = risk_icons.get(result['risk_category'], '⚪')
                st.metric("Risk Category",
                          f"{icon} {result['risk_category']}")
            with m4:
                st.metric("Estimated Premium",
                          f"₹{result['estimated_premium']:,}")

            # Details
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"""
                **Insurance Recommendation**
                - Product: **{result['recommended_product']}**
                - Premium Multiplier: **{result['premium_multiplier']}x**
                - Base ₹12,000 × {result['premium_multiplier']} = **₹{result['estimated_premium']:,}**
                """)

            with col_b:
                # Risk bar chart
                segments = {
                    'Low\n(<10k km)'      : 10000,
                    'Medium\n(10-25k)'    : 15000,
                    'High\n(25-40k)'      : 15000,
                    'Very High\n(40-70k)' : 30000,
                    'Extreme\n(>70k)'     : 50000,
                }
                fig, ax = plt.subplots(figsize=(6, 3))
                colors  = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
                ax.barh(list(segments.keys()), list(segments.values()),
                        color=colors, alpha=0.7)
                ax.axvline(result['predicted_kms'], color='black',
                           linewidth=3, linestyle='--',
                           label=f"Your vehicle: {result['predicted_kms']:,} km")
                ax.set_title("Your Vehicle in Risk Distribution", fontweight='bold')
                ax.legend()
                ax.set_xlabel("Annual KMs")
                st.pyplot(fig)
                plt.close()

            # SHAP explanation
            if result.get('shap_explanation'):
                st.subheader("🔍 Why This Prediction? (SHAP)")
                st.caption("Features that pushed the prediction up (+) or down (-)")
                shap_data = result['shap_explanation']
                features  = list(shap_data.keys())
                values    = list(shap_data.values())
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                colors2   = ['#1F4E79' if v > 0 else '#C00000' for v in values]
                ax2.barh(features[::-1], values[::-1], color=colors2[::-1], alpha=0.85)
                ax2.axvline(0, color='black', linewidth=1)
                ax2.set_title("SHAP Feature Impact\nBlue = pushes km UP | Red = pushes km DOWN",
                              fontweight='bold')
                ax2.set_xlabel("SHAP Value (km impact)")
                st.pyplot(fig2)
                plt.close()

            st.caption(f"Model: {result['model_info']['model_name']} | "
                       f"MAPE: {result['model_info']['mape']}% | "
                       f"Trained: {result['model_info']['trained_on']}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("DV Analytics | Vehicle Mileage MLOps | Author: Suresh D R")
