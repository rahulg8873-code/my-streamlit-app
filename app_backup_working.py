import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# GEOPOLYMER AI
# PyArrow-free prototype
# =========================================================

st.set_page_config(
    page_title="GeoPolymer AI",
    page_icon="🏗️",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("🏗️ GeoPolymer AI")
st.subheader("Geopolymer Concrete Mix Design & Strength Prediction")

st.info(
    "GGBS-based geopolymer concrete research prototype. "
    "All major mix parameters are editable."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Mix Parameters")

grade = st.sidebar.selectbox(
    "Target Concrete Grade",
    ["M30", "M35", "M40", "M45", "M50"],
    index=2
)

age = st.sidebar.selectbox(
    "Testing Age (days)",
    [7, 14, 28, 56, 90],
    index=2
)

precursor = st.sidebar.selectbox(
    "Main Precursor",
    ["GGBS", "Fly Ash", "GGBS + Fly Ash"],
    index=0
)

precursor_mass = st.sidebar.number_input(
    "Precursor / Binder (kg/m³)",
    min_value=100.0,
    max_value=800.0,
    value=400.0,
    step=10.0
)

activator_ratio = st.sidebar.number_input(
    "Activator / Precursor Ratio",
    min_value=0.20,
    max_value=1.00,
    value=0.40,
    step=0.01
)

water_binder = st.sidebar.number_input(
    "Water / Binder Ratio",
    min_value=0.15,
    max_value=0.80,
    value=0.35,
    step=0.01
)

fine_aggregate = st.sidebar.number_input(
    "Fine Aggregate (kg/m³)",
    min_value=400.0,
    max_value=1200.0,
    value=700.0,
    step=10.0
)

coarse_aggregate = st.sidebar.number_input(
    "Coarse Aggregate (kg/m³)",
    min_value=600.0,
    max_value=1600.0,
    value=1100.0,
    step=10.0
)

# =========================================================
# CALCULATIONS
# =========================================================

activator_mass = precursor_mass * activator_ratio
water_content = precursor_mass * water_binder

total_material = (
    precursor_mass
    + activator_mass
    + water_content
    + fine_aggregate
    + coarse_aggregate
)

target_strength = {
    "M30": 30,
    "M35": 35,
    "M40": 40,
    "M45": 45,
    "M50": 50
}[grade]

age_factor = {
    7: 0.65,
    14: 0.82,
    28: 1.00,
    56: 1.08,
    90: 1.12
}[age]

estimated_strength = target_strength * age_factor

# =========================================================
# MIX DESIGN SUMMARY
# =========================================================

st.header("📊 Mix Design Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Precursor",
    f"{precursor_mass:.0f} kg/m³"
)

c2.metric(
    "Activator",
    f"{activator_mass:.0f} kg/m³"
)

c3.metric(
    "Water",
    f"{water_content:.0f} kg/m³"
)

c4.metric(
    "Estimated Strength",
    f"{estimated_strength:.1f} MPa"
)

# =========================================================
# PROPOSED MIX COMPOSITION
# =========================================================

st.header("🧱 Proposed Mix Composition")

mix_data = pd.DataFrame({
    "Material": [
        "Precursor / Binder",
        "Alkaline Activator",
        "Water",
        "Fine Aggregate",
        "Coarse Aggregate"
    ],
    "Quantity (kg/m³)": [
        precursor_mass,
        activator_mass,
        water_content,
        fine_aggregate,
        coarse_aggregate
    ]
})

st.markdown(
    mix_data.to_html(index=False),
    unsafe_allow_html=True
)

# =========================================================
# MATERIAL & MIX PARAMETERS
# =========================================================

st.header("🔬 Material & Mix Parameters")

parameter_data = pd.DataFrame({
    "Parameter": [
        "Concrete Grade",
        "Main Precursor",
        "Testing Age",
        "Activator / Precursor",
        "Water / Binder",
        "Total Material"
    ],
    "Value": [
        grade,
        precursor,
        f"{age} days",
        f"{activator_ratio:.2f}",
        f"{water_binder:.2f}",
        f"{total_material:.0f} kg/m³"
    ]
})

st.markdown(
    parameter_data.to_html(index=False),
    unsafe_allow_html=True
)

# =========================================================
# STRENGTH DEVELOPMENT
# =========================================================

st.header("📈 Estimated Strength Development")

ages = [7, 14, 28, 56, 90]

strength_values = [
    target_strength * 0.65,
    target_strength * 0.82,
    target_strength * 1.00,
    target_strength * 1.08,
    target_strength * 1.12
]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=ages,
        y=strength_values,
        mode="lines+markers",
        name="Estimated Strength"
    )
)

fig.add_hline(
    y=target_strength,
    line_dash="dash",
    annotation_text=f"Target {grade}"
)

fig.update_layout(
    title="Strength vs Curing Age",
    xaxis_title="Curing Age (days)",
    yaxis_title="Compressive Strength (MPa)",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# LAB DATA
# =========================================================

st.header("🧪 Experimental / Lab Data")

st.write(
    "Enter actual laboratory compressive-strength results. "
    "These values will later be used for machine-learning training."
)

lab_data = pd.DataFrame({
    "Age (days)": [7, 14, 28],
    "Actual Strength (MPa)": [0.0, 0.0, 0.0]
})

st.markdown(
    lab_data.to_html(index=False),
    unsafe_allow_html=True
)

if st.button("💾 Save Lab Data"):
    st.success(
        "Lab data captured successfully for this session."
    )

# =========================================================
# AI / MACHINE LEARNING
# =========================================================

st.header("🤖 AI / Machine Learning")

st.warning(
    "The current strength value is an engineering estimate. "
    "It is NOT a trained machine-learning prediction yet."
)

ml_models = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "Support Vector Machine",
        "Artificial Neural Network",
        "LSTM"
    ],
    "Status": [
        "Planned",
        "Primary Model",
        "Planned",
        "Planned",
        "Planned"
    ]
})

st.markdown(
    ml_models.to_html(index=False),
    unsafe_allow_html=True
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "GeoPolymer AI — Research Prototype | "
    "All final mix designs and predictions must be validated "
    "through laboratory testing."
)