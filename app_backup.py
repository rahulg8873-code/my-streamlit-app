import streamlit as st

# ============================================================
# GEOPOLYMER AI - CONFIGURABLE SOFTWARE FRAMEWORK
# ============================================================

st.set_page_config(
    page_title="GeoPolymer AI",
    page_icon="🏗️",
    layout="wide"
)

# -----------------------------
# DEFAULT PROJECT SETTINGS
# -----------------------------

DEFAULT_SETTINGS = {
    "precursor": "GGBS",
    "ggbs_percent": 100.0,
    "fly_ash_percent": 0.0,

    "naoh_molarity": 10.0,
    "na2sio3_naoh_ratio": 2.0,
    "activator_binder_ratio": 0.40,
    "water_binder_ratio": 0.40,

    "ggbs_quantity": 450.0,
    "fine_aggregate": 700.0,
    "coarse_aggregate": 1100.0,
    "water": 0.0,
    "superplasticizer": 0.0,

    "curing_temperature": 25.0,
    "curing_duration": 24.0,
    "curing_method": "Ambient",

    "target_strength": 40.0,

    "ggbs_rate": 0.0,
    "naoh_rate": 0.0,
    "na2sio3_rate": 0.0,
    "fine_aggregate_rate": 0.0,
    "coarse_aggregate_rate": 0.0,
    "sp_rate": 0.0
}

# Store settings in session
if "settings" not in st.session_state:
    st.session_state.settings = DEFAULT_SETTINGS.copy()

settings = st.session_state.settings


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏗️ GeoPolymer AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧪 Mix Designer",
        "📊 Experimental Data",
        "🤖 Strength Prediction",
        "🧠 ML Models",
        "🎯 Mix Optimization",
        "💰 Cost Analysis",
        "🌱 CO₂ Analysis",
        "🔬 Crack Detection",
        "⭐ Quality Score",
        "💡 Smart Recommendation",
        "📄 Generate Report",
        "⚙️ Settings"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🏗️ GeoPolymer AI")
    st.subheader("100% GGBS-Based Geopolymer Concrete")

    st.markdown(
        """
        ### AI-Based Mix Design & Performance Prediction Software

        This software framework is designed for:

        - Geopolymer concrete mix design
        - Experimental data management
        - ML-based strength prediction
        - Mix optimization
        - Cost analysis
        - CO₂ analysis
        - Experimental validation
        - Automatic reporting
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precursor",
        settings["precursor"]
    )

    col2.metric(
        "GGBS %",
        f'{settings["ggbs_percent"]:.1f}%'
    )

    col3.metric(
        "Target Strength",
        f'{settings["target_strength"]:.1f} MPa'
    )

    col4.metric(
        "Curing",
        settings["curing_method"]
    )

    st.info(
        "⚙️ Project parameters can be changed from the Settings module."
    )


# ============================================================
# MIX DESIGNER
# ============================================================

elif page == "🧪 Mix Designer":

    st.title("🧪 Geopolymer Mix Designer")

    st.write(
        "Enter experimental mix parameters. "
        "All values are configurable from the software."
    )

    st.subheader("Binder")

    col1, col2 = st.columns(2)

    with col1:
        ggbs = st.number_input(
            "GGBS (kg/m³)",
            min_value=0.0,
            value=float(settings["ggbs_quantity"]),
            step=1.0
        )

    with col2:
        fly_ash = st.number_input(
            "Fly Ash (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(settings["fly_ash_percent"]),
            step=1.0
        )

    st.subheader("Aggregates")

    col1, col2 = st.columns(2)

    with col1:
        fine_aggregate = st.number_input(
            "Fine Aggregate (kg/m³)",
            min_value=0.0,
            value=float(settings["fine_aggregate"]),
            step=1.0
        )

    with col2:
        coarse_aggregate = st.number_input(
            "Coarse Aggregate (kg/m³)",
            min_value=0.0,
            value=float(settings["coarse_aggregate"]),
            step=1.0
        )

    st.subheader("Alkaline Activator")

    col1, col2 = st.columns(2)

    with col1:
        naoh = st.number_input(
            "NaOH (kg/m³)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        naoh_molarity = st.number_input(
            "NaOH Molarity (M)",
            min_value=0.0,
            value=float(settings["naoh_molarity"]),
            step=0.5
        )

    with col2:
        na2sio3 = st.number_input(
            "Na₂SiO₃ (kg/m³)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        activator_ratio = st.number_input(
            "Na₂SiO₃ / NaOH Ratio",
            min_value=0.0,
            value=float(settings["na2sio3_naoh_ratio"]),
            step=0.1
        )

    st.subheader("Mix Ratios")

    col1, col2, col3 = st.columns(3)

    with col1:
        ab_ratio = st.number_input(
            "Activator / Binder Ratio",
            min_value=0.0,
            value=float(settings["activator_binder_ratio"]),
            step=0.01
        )

    with col2:
        wb_ratio = st.number_input(
            "Water / Binder Ratio",
            min_value=0.0,
            value=float(settings["water_binder_ratio"]),
            step=0.01
        )

    with col3:
        water = st.number_input(
            "Water (kg/m³)",
            min_value=0.0,
            value=float(settings["water"]),
            step=1.0
        )

    st.subheader("Admixture")

    sp = st.number_input(
        "Superplasticizer (kg/m³)",
        min_value=0.0,
        value=float(settings["superplasticizer"]),
        step=0.1
    )

    st.subheader("Curing Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        curing_temperature = st.number_input(
            "Curing Temperature (°C)",
            value=float(settings["curing_temperature"]),
            step=1.0
        )

    with col2:
        curing_duration = st.number_input(
            "Curing Duration (hours)",
            value=float(settings["curing_duration"]),
            step=1.0
        )

    with col3:
        curing_method = st.selectbox(
            "Curing Method",
            [
                "Ambient",
                "Heat Curing",
                "Oven Curing",
                "Steam Curing",
                "Other"
            ]
        )

    target_strength = st.number_input(
        "Target Compressive Strength (MPa)",
        min_value=0.0,
        value=float(settings["target_strength"]),
        step=1.0
    )

    st.divider()

    if st.button("Calculate Mix Summary", type="primary"):

        total_binder = ggbs

        st.success("Mix parameters received successfully.")

        st.subheader("Mix Design Summary")

        summary = {
            "GGBS": f"{ggbs:.2f} kg/m³",
            "Fly Ash": f"{fly_ash:.2f} %",
            "Fine Aggregate": f"{fine_aggregate:.2f} kg/m³",
            "Coarse Aggregate": f"{coarse_aggregate:.2f} kg/m³",
            "NaOH": f"{naoh:.2f} kg/m³",
            "Na₂SiO₃": f"{na2sio3:.2f} kg/m³",
            "NaOH Molarity": f"{naoh_molarity:.2f} M",
            "Na₂SiO₃/NaOH": f"{activator_ratio:.2f}",
            "Activator/Binder": f"{ab_ratio:.2f}",
            "Water/Binder": f"{wb_ratio:.2f}",
            "Water": f"{water:.2f} kg/m³",
            "Superplasticizer": f"{sp:.2f} kg/m³",
            "Curing": f"{curing_method}, {curing_temperature:.1f} °C, {curing_duration:.1f} h",
            "Target Strength": f"{target_strength:.2f} MPa"
        }

        for material, quantity in summary.items():
            st.write(f"**{material}:** {quantity}")

        st.warning(
            "These are user-entered parameters. "
            "Final mix-design equations will be added after the experimental methodology is finalized."
        )


# ============================================================
# EXPERIMENTAL DATA
# ============================================================

elif page == "📊 Experimental Data":

    st.title("📊 Experimental Data")

    st.info(
        "Laboratory results such as 7-day, 28-day and 56-day strength "
        "will be entered/imported here."
    )

    st.subheader("Required Dataset Fields")

    fields = [
        "Mix ID",
        "GGBS",
        "NaOH Molarity",
        "Na₂SiO₃/NaOH Ratio",
        "Activator/Binder Ratio",
        "Water/Binder Ratio",
        "Fine Aggregate",
        "Coarse Aggregate",
        "Superplasticizer",
        "Curing Temperature",
        "Curing Duration",
        "Curing Method",
        "Slump",
        "Density",
        "7-Day Strength",
        "28-Day Strength",
        "56-Day Strength"
    ]

    for field in fields:
        st.write("•", field)


# ============================================================
# STRENGTH PREDICTION
# ============================================================

elif page == "🤖 Strength Prediction":

    st.title("🤖 AI Strength Prediction")

    st.info(
        "ML prediction will be activated after sufficient experimental "
        "training data is available."
    )

    st.write("Planned outputs:")
    st.write("• 7-Day Compressive Strength")
    st.write("• 28-Day Compressive Strength")
    st.write("• 56-Day Compressive Strength")


# ============================================================
# ML MODELS
# ============================================================

elif page == "🧠 ML Models":

    st.title("🧠 Machine Learning Model Comparison")

    st.write("Planned models:")

    models = [
        "Random Forest",
        "XGBoost",
        "Artificial Neural Network",
        "Gradient Boosting",
        "Support Vector Regression"
    ]

    for model in models:
        st.write("•", model)

    st.info(
        "R², RMSE, MAE and MAPE will be calculated after experimental "
        "data is available."
    )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "🎯 Mix Optimization":

    st.title("🎯 Mix Optimization")

    st.info(
        "Optimization will search within user-defined experimental "
        "ranges to identify mixes meeting the target strength."
    )

    st.write("Future optimization objectives:")
    st.write("• Target strength")
    st.write("• Minimum cost")
    st.write("• Lower estimated CO₂")
    st.write("• Experimental constraints")


# ============================================================
# COST
# ============================================================

elif page == "💰 Cost Analysis":

    st.title("💰 Cost Analysis")

    st.info(
        "Material rates will be user-editable from the Settings module."
    )

    st.write("Current rates:")
    st.write(f"GGBS: ₹{settings['ggbs_rate']:.2f}/kg")
    st.write(f"NaOH: ₹{settings['naoh_rate']:.2f}/kg")
    st.write(f"Na₂SiO₃: ₹{settings['na2sio3_rate']:.2f}/kg")


# ============================================================
# CO2
# ============================================================

elif page == "🌱 CO₂ Analysis":

    st.title("🌱 CO₂ Analysis")

    st.info(
        "Emission factors will be user-editable and documented with "
        "their sources before final calculations are activated."
    )


# ============================================================
# CRACK DETECTION
# ============================================================

elif page == "🔬 Crack Detection":

    st.title("🔬 Concrete Crack Detection")

    st.info(
        "Image-based crack detection will be added in the advanced AI phase."
    )

    uploaded = st.file_uploader(
        "Upload specimen image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded specimen")


# ============================================================
# QUALITY SCORE
# ============================================================

elif page == "⭐ Quality Score":

    st.title("⭐ Geopolymer Mix Quality Score")

    st.info(
        "The scoring formula will be transparent and documented after "
        "experimental and economic parameters are finalized."
    )


# ============================================================
# SMART RECOMMENDATION
# ============================================================

elif page == "💡 Smart Recommendation":

    st.title("💡 Smart Recommendation System")

    st.info(
        "Recommendations will be generated using trained model results "
        "and user-defined experimental limits."
    )


# ============================================================
# REPORT
# ============================================================

elif page == "📄 Generate Report":

    st.title("📄 Automatic Project Report")

    st.info(
        "The final PDF report module will combine mix design, ML, "
        "experimental validation, cost and CO₂ results."
    )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Project Settings")

    st.write(
        "All major project parameters are editable here. "
        "Changes will be used by the software framework."
    )

    st.subheader("Binder")

    settings["precursor"] = st.selectbox(
        "Precursor",
        ["GGBS", "Fly Ash", "GGBS + Fly Ash"],
        index=["GGBS", "Fly Ash", "GGBS + Fly Ash"].index(
            settings["precursor"]
        )
    )

    settings["ggbs_percent"] = st.number_input(
        "GGBS Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(settings["ggbs_percent"]),
        step=1.0
    )

    settings["fly_ash_percent"] = st.number_input(
        "Fly Ash Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(settings["fly_ash_percent"]),
        step=1.0
    )

    st.subheader("Activator")

    settings["naoh_molarity"] = st.number_input(
        "NaOH Molarity (M)",
        min_value=0.0,
        value=float(settings["naoh_molarity"]),
        step=0.5
    )

    settings["na2sio3_naoh_ratio"] = st.number_input(
        "Na₂SiO₃ / NaOH Ratio",
        min_value=0.0,
        value=float(settings["na2sio3_naoh_ratio"]),
        step=0.1
    )

    settings["activator_binder_ratio"] = st.number_input(
        "Activator / Binder Ratio",
        min_value=0.0,
        value=float(settings["activator_binder_ratio"]),
        step=0.01
    )

    settings["water_binder_ratio"] = st.number_input(
        "Water / Binder Ratio",
        min_value=0.0,
        value=float(settings["water_binder_ratio"]),
        step=0.01
    )

    st.subheader("Materials")

    settings["ggbs_quantity"] = st.number_input(
        "GGBS Quantity (kg/m³)",
        min_value=0.0,
        value=float(settings["ggbs_quantity"]),
        step=1.0
    )

    settings["fine_aggregate"] = st.number_input(
        "Fine Aggregate (kg/m³)",
        min_value=0.0,
        value=float(settings["fine_aggregate"]),
        step=1.0
    )

    settings["coarse_aggregate"] = st.number_input(
        "Coarse Aggregate (kg/m³)",
        min_value=0.0,
        value=float(settings["coarse_aggregate"]),
        step=1.0
    )

    settings["superplasticizer"] = st.number_input(
        "Superplasticizer (kg/m³)",
        min_value=0.0,
        value=float(settings["superplasticizer"]),
        step=0.1
    )

    st.subheader("Curing")

    settings["curing_temperature"] = st.number_input(
        "Curing Temperature (°C)",
        value=float(settings["curing_temperature"]),
        step=1.0
    )

    settings["curing_duration"] = st.number_input(
        "Curing Duration (hours)",
        min_value=0.0,
        value=float(settings["curing_duration"]),
        step=1.0
    )

    settings["curing_method"] = st.selectbox(
        "Curing Method",
        [
            "Ambient",
            "Heat Curing",
            "Oven Curing",
            "Steam Curing",
            "Other"
        ],
        index=[
            "Ambient",
            "Heat Curing",
            "Oven Curing",
            "Steam Curing",
            "Other"
        ].index(settings["curing_method"])
    )

    st.subheader("Target")

    settings["target_strength"] = st.number_input(
        "Target Compressive Strength (MPa)",
        min_value=0.0,
        value=float(settings["target_strength"]),
        step=1.0
    )

    st.subheader("Material Cost Rates")

    settings["ggbs_rate"] = st.number_input(
        "GGBS Rate (₹/kg)",
        min_value=0.0,
        value=float(settings["ggbs_rate"]),
        step=0.1
    )

    settings["naoh_rate"] = st.number_input(
        "NaOH Rate (₹/kg)",
        min_value=0.0,
        value=float(settings["naoh_rate"]),
        step=0.1
    )

    settings["na2sio3_rate"] = st.number_input(
        "Na₂SiO₃ Rate (₹/kg)",
        min_value=0.0,
        value=float(settings["na2sio3_rate"]),
        step=0.1
    )

    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings updated for this session.")


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("GeoPolymer AI | Configurable Research Framework")