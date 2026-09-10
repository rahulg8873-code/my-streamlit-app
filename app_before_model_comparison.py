import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="GeoPolymer AI",
    page_icon="🧱",
    layout="wide"
)


# =========================================================
# BASIC SETTINGS
# =========================================================

TARGET = "28-Day Strength (MPa)"

FEATURES = [
    "GGBS (kg/m³)",
    "NaOH Molarity (M)",
    "Na₂SiO₃/NaOH Ratio",
    "Activator/Binder Ratio",
    "Water/Binder Ratio",
    "Fine Aggregate (kg/m³)",
    "Coarse Aggregate (kg/m³)",
    "Superplasticizer (kg/m³)",
    "Curing Temperature (°C)",
    "Curing Duration (h)",
    "Curing Method",
    "Slump (mm)",
    "Density (kg/m³)"
]


# =========================================================
# SESSION STATE
# =========================================================

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""

if "training_result" not in st.session_state:
    st.session_state.training_result = None


# =========================================================
# COLUMN NORMALIZATION
# =========================================================

def normalize_columns(df):

    rename = {}

    for column in df.columns:

        text = str(column).strip().lower()

        if "ggbs" in text:
            rename[column] = "GGBS (kg/m³)"

        elif "naoh" in text and ("molar" in text or "modality" in text):
            rename[column] = "NaOH Molarity (M)"

        elif "na2sio3" in text or ("sio3" in text and "naoh" in text):
            rename[column] = "Na₂SiO₃/NaOH Ratio"

        elif "activator" in text and "binder" in text:
            rename[column] = "Activator/Binder Ratio"

        elif "water" in text and "binder" in text:
            rename[column] = "Water/Binder Ratio"

        elif "fine aggregate" in text:
            rename[column] = "Fine Aggregate (kg/m³)"

        elif "coarse aggregate" in text:
            rename[column] = "Coarse Aggregate (kg/m³)"

        elif "superplasticizer" in text:
            rename[column] = "Superplasticizer (kg/m³)"

        elif "curing temperature" in text:
            rename[column] = "Curing Temperature (°C)"

        elif "curing duration" in text:
            rename[column] = "Curing Duration (h)"

        elif "curing method" in text:
            rename[column] = "Curing Method"

        elif "slump" in text:
            rename[column] = "Slump (mm)"

        elif "density" in text:
            rename[column] = "Density (kg/m³)"

        elif "7" in text and "strength" in text:
            rename[column] = "7-Day Strength (MPa)"

        elif "28" in text and "strength" in text:
            rename[column] = "28-Day Strength (MPa)"

        elif "56" in text and "strength" in text:
            rename[column] = "56-Day Strength (MPa)"

        elif "mix" in text and "id" in text:
            rename[column] = "Mix ID"

    return df.rename(columns=rename)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(uploaded_file):

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        return normalize_columns(df)

    else:

        excel_file = pd.ExcelFile(uploaded_file)

        sheets = pd.read_excel(
            uploaded_file,
            sheet_name=None
        )

        for sheet_name, sheet_df in sheets.items():

            normalized = normalize_columns(sheet_df)

            if TARGET in normalized.columns:

                return normalized

        first_sheet = list(sheets.keys())[0]

        return normalize_columns(
            sheets[first_sheet]
        )


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    data = df.copy()

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        return None, missing

    data = data[required].copy()

    numeric_columns = [
        column
        for column in FEATURES
        if column != "Curing Method"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce"
    )

    data = data.dropna(
        subset=[TARGET]
    )

    return data, []


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_model(
    df,
    train_percentage,
    trees,
    depth,
    learning_rate
):

    data, missing = prepare_data(df)

    if data is None:

        return None, missing

    if len(data) < 10:

        return "NOT_ENOUGH_DATA", []

    X = data[FEATURES]

    y = data[TARGET]

    numeric_features = [
        column
        for column in FEATURES
        if column != "Curing Method"
    ]

    categorical_features = [
        "Curing Method"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = XGBRegressor(

        n_estimators=int(trees),

        max_depth=int(depth),

        learning_rate=float(
            learning_rate
        ),

        subsample=0.9,

        colsample_bytree=0.9,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=1
    )

    pipeline = Pipeline(

        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        train_size=train_percentage / 100,

        random_state=42
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    result = {

        "pipeline": pipeline,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": predictions,

        "R2": r2,

        "MAE": mae,

        "RMSE": rmse,

        "train_records": len(X_train),

        "test_records": len(X_test)
    }

    return result, []


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🧱 GeoPolymer AI"
)

st.sidebar.write(
    "Geopolymer Concrete Mix Design and Strength Prediction"
)

st.sidebar.divider()

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Mix Design",
        "Activator",
        "Mix Parameters",
        "Strength Prediction",
        "Experimental Data",
        "AI / XGBoost"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "Dataset Mode"
)

dataset_mode = st.sidebar.radio(

    "Select Dataset",

    [
        "Demo / Synthetic",
        "Real Laboratory"
    ]
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🧱 GeoPolymer AI"
)

st.subheader(
    "Geopolymer Concrete Mix Design and Strength Prediction"
)

st.caption(
    "Research and prototype platform"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset Mode",
            dataset_mode
        )

    with col2:

        if st.session_state.uploaded_df is None:

            records = 0

        else:

            records = len(
                st.session_state.uploaded_df
            )

        st.metric(
            "Records",
            records
        )

    with col3:

        st.metric(
            "Primary Model",
            "XGBoost"
        )

    with col4:

        st.metric(
            "ML Target",
            "28-Day Strength"
        )

    st.divider()

    st.subheader(
        "Project Configuration"
    )

    st.success(
        "Precursor: 100% GGBS"
    )

    st.info(
        "Fly ash is currently not being used."
    )

    st.write(
        "Prototype target strength: 40 MPa"
    )

    st.divider()

    st.subheader(
        "AI Research Workflow"
    )

    st.write(
        "Dataset → Data Validation → Feature Selection → "
        "Training/Testing → XGBoost → Strength Prediction → "
        "Model Evaluation"
    )


# =========================================================
# MIX DESIGN
# =========================================================

elif page == "Mix Design":

    st.header(
        "Mix Design"
    )

    st.success(
        "Current precursor configuration: 100% GGBS"
    )

    st.number_input(
        "GGBS Percentage (%)",
        min_value=0,
        max_value=100,
        value=100
    )

    st.number_input(
        "Fly Ash Percentage (%)",
        min_value=0,
        max_value=100,
        value=0
    )

    st.number_input(
        "Target Strength (MPa)",
        min_value=10.0,
        max_value=100.0,
        value=40.0
    )


# =========================================================
# ACTIVATOR
# =========================================================

elif page == "Activator":

    st.header(
        "Activator"
    )

    st.number_input(
        "NaOH Molarity (M)",
        value=12.0
    )

    st.number_input(
        "Na₂SiO₃ / NaOH Ratio",
        value=2.0
    )

    st.number_input(
        "Activator / Binder Ratio",
        value=0.45
    )

    st.success(
        "Activator parameters are ready."
    )


# =========================================================
# MIX PARAMETERS
# =========================================================

elif page == "Mix Parameters":

    st.header(
        "Mix Parameters"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.number_input(
            "GGBS (kg/m³)",
            value=400.0
        )

        st.number_input(
            "Precursor Percentage (%)",
            value=100.0
        )

    with col2:

        st.number_input(
            "Activator / GGBS Ratio",
            value=0.45
        )

        st.number_input(
            "NaOH Molarity",
            value=12.0
        )

    with col3:

        st.number_input(
            "Silicate / NaOH",
            value=2.0
        )

        st.number_input(
            "Water / Binder Ratio",
            value=0.35
        )


# =========================================================
# STRENGTH PREDICTION
# =========================================================

elif page == "Strength Prediction":

    st.header(
        "Strength Prediction"
    )

    st.write(
        "Curing Age: 28 days"
    )

    st.write(
        "Prototype Estimation: 40 MPa"
    )

    if st.session_state.training_result is None:

        st.warning(
            "Train XGBoost from the AI / XGBoost section "
            "to activate ML-based prediction."
        )

    else:

        st.success(
            "XGBoost model is trained."
        )

        result = (
            st.session_state.training_result
        )

        st.write(
            "Model R²:",
            round(result["R2"], 4)
        )

        st.write(
            "Model MAE:",
            round(result["MAE"], 3),
            "MPa"
        )

        st.write(
            "Model RMSE:",
            round(result["RMSE"], 3),
            "MPa"
        )


# =========================================================
# EXPERIMENTAL DATA
# =========================================================

elif page == "Experimental Data":

    st.header(
        "Experimental / Laboratory Data"
    )

    if dataset_mode == "Real Laboratory":

        st.info(
            "This section is reserved for actual laboratory measurements."
        )

    else:

        st.info(
            "Demo/Synthetic data is being used for ML workflow testing."
        )

    if st.session_state.uploaded_df is None:

        st.warning(
            "No dataset uploaded yet."
        )

    else:

        st.dataframe(
            st.session_state.uploaded_df,
            use_container_width=True
        )


# =========================================================
# AI / XGBOOST
# =========================================================

elif page == "AI / XGBoost":

    st.header(
        "🤖 AI / Machine Learning"
    )

    st.write(
        "Primary Model: **XGBoost**"
    )

    st.caption(
        "Geopolymer AI research workflow"
    )

    # -----------------------------------------------------
    # DATASET UPLOAD
    # -----------------------------------------------------

    st.subheader(
        "Experimental Dataset Upload"
    )

    uploaded_file = st.file_uploader(

        "Upload your experimental Excel / CSV dataset",

        type=[
            "xlsx",
            "xls",
            "csv"
        ]
    )

    if uploaded_file is not None:

        try:

            dataframe = load_dataset(
                uploaded_file
            )

            st.session_state.uploaded_df = dataframe

            st.session_state.dataset_name = (
                uploaded_file.name
            )

            st.session_state.training_result = None

            st.success(
                "Dataset loaded successfully."
            )

        except Exception as error:

            st.error(
                f"Dataset loading error: {error}"
            )


    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    if st.session_state.uploaded_df is not None:

        df = (
            st.session_state.uploaded_df
        )

        st.divider()

        st.subheader(
            "Dataset Information"
        )

        st.write(
            "Dataset:",
            st.session_state.dataset_name
        )

        st.write(
            "Total Records:",
            len(df)
        )

        # Missing values

        st.write(
            "Missing Values by Column"
        )

        missing_values = df.isna().sum()

        missing_table = pd.DataFrame({

            "Column":
                missing_values.index,

            "Missing Values":
                missing_values.values
        })

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

        if missing_values.sum() == 0:

            st.success(
                "No missing values detected."
            )


        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        st.subheader(
            "ML Target"
        )

        st.success(
            "Primary Target: 28-Day Strength (MPa)"
        )


        # -------------------------------------------------
        # FEATURE SELECTION
        # -------------------------------------------------

        st.subheader(
            "Feature Selection"
        )

        available_features = [

            feature

            for feature in FEATURES

            if feature in df.columns
        ]

        missing_features = [

            feature

            for feature in FEATURES

            if feature not in df.columns
        ]

        if missing_features:

            st.error(
                "Missing Input Features: "
                + ", ".join(
                    missing_features
                )
            )

        else:

            st.success(
                "All required input features are available."
            )

            st.write(
                "Input Features:"
            )

            st.write(
                ", ".join(
                    available_features
                )
            )


        # -------------------------------------------------
        # TRAIN / TEST
        # -------------------------------------------------

        st.subheader(
            "Training / Testing"
        )

        train_percentage = st.slider(

            "Training Data Percentage",

            min_value=50,

            max_value=90,

            value=80,

            step=5
        )

        test_percentage = (
            100 - train_percentage
        )

        train_records = int(
            len(df)
            * train_percentage
            / 100
        )

        test_records = (
            len(df)
            - train_records
        )

        st.write(
            f"Training Data: **{train_percentage}%**"
        )

        st.write(
            f"Testing Data: **{test_percentage}%**"
        )

        st.write(
            "Random State: **42**"
        )

        st.write(
            f"Training Records: **{train_records}**"
        )

        st.write(
            f"Testing Records: **{test_records}**"
        )


        # -------------------------------------------------
        # XGBOOST SETTINGS
        # -------------------------------------------------

        st.subheader(
            "XGBoost Model Parameters"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            trees = st.number_input(

                "Number of Trees",

                min_value=50,

                max_value=1000,

                value=200,

                step=50
            )

        with col2:

            depth = st.number_input(

                "Maximum Depth",

                min_value=2,

                max_value=12,

                value=4,

                step=1
            )

        with col3:

            learning_rate = st.number_input(

                "Learning Rate",

                min_value=0.001,

                max_value=1.0,

                value=0.05,

                step=0.01
            )


        # -------------------------------------------------
        # TRAIN BUTTON
        # -------------------------------------------------

        st.divider()

        if missing_features:

            st.error(
                "Training cannot start until all input features are available."
            )

        else:

            train_button = st.button(

                "🚀 Train XGBoost Model",

                type="primary",

                use_container_width=True
            )

            if train_button:

                with st.spinner(
                    "Training XGBoost model..."
                ):

                    result, errors = train_model(

                        df,

                        train_percentage,

                        trees,

                        depth,

                        learning_rate
                    )

                if errors:

                    st.error(
                        "Training error: "
                        + ", ".join(
                            errors
                        )
                    )

                elif result == "NOT_ENOUGH_DATA":

                    st.error(
                        "Not enough usable records for training."
                    )

                else:

                    st.session_state.training_result = result

                    st.success(
                        "XGBoost training completed successfully!"
                    )


        # -------------------------------------------------
        # MODEL RESULTS
        # -------------------------------------------------

        if (
            st.session_state.training_result
            is not None
        ):

            result = (
                st.session_state.training_result
            )

            st.divider()

            st.subheader(
                "📊 Model Performance"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "R²",
                    f"{result['R2']:.4f}"
                )

            with col2:

                st.metric(
                    "MAE",
                    f"{result['MAE']:.3f} MPa"
                )

            with col3:

                st.metric(
                    "RMSE",
                    f"{result['RMSE']:.3f} MPa"
                )

            with col4:

                st.metric(
                    "Testing Records",
                    result["test_records"]
                )


            # ---------------------------------------------
            # ACTUAL VS PREDICTED
            # ---------------------------------------------

            st.subheader(
                "Actual vs Predicted — 28-Day Strength"
            )

            comparison = pd.DataFrame({

                "Actual Strength (MPa)":
                    result["y_test"].values,

                "Predicted Strength (MPa)":
                    result["predictions"]
            })

            st.dataframe(

                comparison.round(3),

                use_container_width=True,

                hide_index=True
            )


            # ---------------------------------------------
            # GRAPH
            # ---------------------------------------------

            figure, axis = plt.subplots()

            axis.scatter(

                comparison[
                    "Actual Strength (MPa)"
                ],

                comparison[
                    "Predicted Strength (MPa)"
                ]
            )

            minimum = min(
                comparison.min()
            )

            maximum = max(
                comparison.max()
            )

            axis.plot(

                [minimum, maximum],

                [minimum, maximum]
            )

            axis.set_xlabel(
                "Actual 28-Day Strength (MPa)"
            )

            axis.set_ylabel(
                "Predicted 28-Day Strength (MPa)"
            )

            axis.set_title(
                "XGBoost Actual vs Predicted"
            )

            st.pyplot(
                figure
            )

            plt.close(
                figure
            )


    # -----------------------------------------------------
    # MODEL FRAMEWORK
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "AI Model Framework"
    )

    framework = pd.DataFrame({

        "Model": [

            "XGBoost",

            "Random Forest",

            "Support Vector Machine",

            "Artificial Neural Network",

            "LSTM"
        ],

        "Role": [

            "Primary Model",

            "Comparison",

            "Comparison",

            "Comparison",

            "Advanced"
        ],

        "Status": [

            "Active",

            "Comparison Planned",

            "Comparison Planned",

            "Comparison Planned",

            "Advanced Planned"
        ]
    })

    st.dataframe(

        framework,

        use_container_width=True,

        hide_index=True
    )


    st.caption(
        "GeoPolymer AI research and mix design and strength prediction platform."
    )