import io
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="GeoPolymer AI",
    page_icon="🏗️",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

FEATURE_CANDIDATES = [
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
    "Slump (mm)",
    "Density (kg/m³)"
]

TARGET_OPTIONS = [
    "7-Day Strength (MPa)",
    "28-Day Strength (MPa)",
    "56-Day Strength (MPa)"
]

STATE_FILE = Path(__file__).with_name("geopolymer_ai_v2_project_state.json")
MODEL_FILE = Path(__file__).with_name("geopolymer_ai_v2_trained_models.pkl")
DATASET_FILE = Path(__file__).with_name("geopolymer_ai_v2_training_dataset.csv")


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

defaults = {
    "ggbs": 400.0,
    "activator_ratio": 0.40,
    "naoh_molarity": 10.0,
    "silicate_naoh_ratio": 2.0,
    "water_binder": 0.35,
    "fine_aggregate": 700.0,
    "coarse_aggregate": 1100.0,
    "superplasticizer": 1.0,
    "curing_temperature": 40.0,
    "curing_duration": 24.0,
    "slump": 100.0,
    "density": 2350.0,

    "ml_results": None,
    "ml_predictions": None,
    "ml_models": {},
    "ml_features": [],
    "ml_target": "28-Day Strength (MPa)",
    "ml_dataset": None,
    "ml_dataset_name": "",
    "ml_dataset_hash": "",
    "ml_restore_status": "",
    "ml_train_X": None,
    "ml_train_y": None,
    "ml_full_X": None,
    "ml_full_y": None,

    "optimization_target": 40.0,
    "optimization_target_met": False,
    "recommended_mix": None,
    "recommended_mix_label": "Primary Recommended Mix",
    "optimization_results": None,
    "optimization_closest_mix": None,
    "optimization_higher_mix": None,
    "optimization_target_met_closest": False,
    "optimization_target_met_higher": False,

    "cost_rates": {
        "GGBS": 8.0,
        "NaOH": 55.0,
        "Sodium Silicate": 25.0,
        "Fine Aggregate": 1.50,
        "Coarse Aggregate": 1.80,
        "Superplasticizer": 120.0
    },
    "cost_result": None,
    "co2_factors": {
        "GGBS": 0.10,
        "NaOH": 1.20,
        "Sodium Silicate": 1.00,
        "Fine Aggregate": 0.005,
        "Coarse Aggregate": 0.004,
        "Superplasticizer": 1.80
    },
    "co2_result": None,
    "quality_result": None,
    "smart_result": None,

    "crack_score": None,
    "crack_area": None,
    "crack_screening": None,
    "crack_severity": None,
    "crack_image_name": "",

    "demo_uploaded_name": "",
    "real_lab_uploaded_name": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



# =========================================================
# ROBUST PROJECT STATE / REPORT HELPERS
# =========================================================

def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return {"__type__": "dataframe", "data": value.to_dict(orient="records")}
    if isinstance(value, pd.Series):
        return {"__type__": "series", "data": value.to_dict()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _restore_json(value):
    if isinstance(value, dict):
        if value.get("__type__") == "dataframe":
            return pd.DataFrame(value.get("data", []))
        if value.get("__type__") == "series":
            return pd.Series(value.get("data", {}))
        return {k: _restore_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_json(v) for v in value]
    return value


def save_project_state():
    keys = [
        "ggbs", "activator_ratio", "naoh_molarity", "silicate_naoh_ratio",
        "water_binder", "fine_aggregate", "coarse_aggregate",
        "superplasticizer", "curing_temperature", "curing_duration",
        "slump", "density", "ml_results", "ml_predictions",
        "ml_features", "ml_target", "ml_dataset_name", "ml_dataset_hash",
        "optimization_target", "optimization_target_met",
        "recommended_mix", "recommended_mix_label", "optimization_results",
        "optimization_closest_mix", "optimization_higher_mix",
        "optimization_target_met_closest", "optimization_target_met_higher",
        "cost_rates", "cost_result", "co2_factors", "co2_result",
        "quality_result", "smart_result", "crack_score", "crack_area",
        "crack_screening", "crack_severity", "crack_image_name",
        "demo_uploaded_name", "real_lab_uploaded_name"
    ]
    state = {
        "project": "GeoPolymer AI",
        "version": "MASTER_STATE_V3",
        "saved_at": datetime.now().isoformat(timespec="seconds")
    }
    for key in keys:
        if key in st.session_state:
            state[key] = _json_safe(st.session_state[key])
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"GeoPolymer AI state save warning: {exc}")


def load_project_state():
    if not STATE_FILE.exists():
        return False
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if state.get("project") != "GeoPolymer AI":
            return False
        for key, value in state.items():
            if key in {"project", "version", "saved_at"}:
                continue
            st.session_state[key] = _restore_json(value)
        return True
    except Exception as exc:
        print(f"GeoPolymer AI state load warning: {exc}")
        return False


def _file_sha256(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def save_model_bundle():
    """Persist trained ML models plus a CSV fallback for restart recovery."""
    try:
        bundle = {
            "version": "ML_BUNDLE_V1",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_name": st.session_state.get("ml_dataset_name", ""),
            "dataset_hash": st.session_state.get("ml_dataset_hash", ""),
            "target": st.session_state.get("ml_target"),
            "features": st.session_state.get("ml_features", []),
            "models": st.session_state.get("ml_models", {}),
            "results": st.session_state.get("ml_results"),
            "predictions": st.session_state.get("ml_predictions"),
            "train_X": st.session_state.get("ml_train_X"),
            "train_y": st.session_state.get("ml_train_y"),
            "full_X": st.session_state.get("ml_full_X"),
            "full_y": st.session_state.get("ml_full_y"),
            "dataset": st.session_state.get("ml_dataset")
        }
        # Always save a human-readable training-data fallback first.
        dataset = bundle.get("dataset")
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            try:
                dataset.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")
            except Exception as dataset_exc:
                print(f"GeoPolymer AI dataset save warning: {dataset_exc}")

        tmp_file = MODEL_FILE.with_suffix(".tmp")
        with tmp_file.open("wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_file.replace(MODEL_FILE)
        st.session_state.ml_restore_status = (
            "Trained models saved successfully. They will be restored after restart."
        )
        return True
    except Exception as exc:
        st.session_state.ml_restore_status = (
            "Model file could not be saved; the training dataset fallback was saved "
            "and will be used for automatic retraining after restart."
        )
        print(f"GeoPolymer AI model save warning: {exc}")
        return False


def load_model_bundle():
    """Restore trained ML models after a Streamlit/server restart."""
    if not MODEL_FILE.exists():
        return False
    try:
        with MODEL_FILE.open("rb") as fh:
            bundle = pickle.load(fh)

        models = bundle.get("models", {})
        features = bundle.get("features", [])
        if not isinstance(models, dict) or not models or not features:
            return False

        st.session_state.ml_models = models
        st.session_state.ml_features = list(features)
        st.session_state.ml_target = bundle.get(
            "target", st.session_state.get("ml_target", "28-Day Strength (MPa)")
        )
        st.session_state.ml_dataset_name = bundle.get(
            "dataset_name", st.session_state.get("ml_dataset_name", "")
        )
        st.session_state.ml_dataset_hash = bundle.get(
            "dataset_hash", st.session_state.get("ml_dataset_hash", "")
        )
        st.session_state.ml_results = bundle.get("results")
        st.session_state.ml_predictions = bundle.get("predictions")
        st.session_state.ml_train_X = bundle.get("train_X")
        st.session_state.ml_train_y = bundle.get("train_y")
        st.session_state.ml_full_X = bundle.get("full_X")
        st.session_state.ml_full_y = bundle.get("full_y")
        st.session_state.ml_dataset = bundle.get("dataset")
        st.session_state.ml_restore_status = (
            "Previously trained models restored automatically from disk."
        )
        return True
    except Exception as exc:
        st.session_state.ml_restore_status = (
            "Saved model bundle could not be loaded. Automatic retraining fallback will be attempted."
        )
        print(f"GeoPolymer AI model load warning: {exc}")
        return False


def auto_recover_trained_models():
    """Recover trained models after restart, even when pickle is unavailable.

    Priority:
    1. Load the persisted pickle model bundle.
    2. If that fails, rebuild the models from the persisted CSV dataset.
    """
    if st.session_state.get("ml_models"):
        return True

    if not DATASET_FILE.exists():
        return False

    try:
        df = pd.read_csv(DATASET_FILE)
        if df.empty:
            return False

        saved_target = st.session_state.get("ml_target", "28-Day Strength (MPa)")
        if saved_target not in df.columns:
            available = [c for c in TARGET_OPTIONS if c in df.columns]
            if not available:
                return False
            saved_target = available[0]
        st.session_state.ml_target = saved_target

        (
            trained_target, features, X_train, X_test, y_train, y_test,
            results, pred_table, trained_models, X, y
        ) = train_model_bundle(df)

        st.session_state.ml_dataset_name = st.session_state.get(
            "ml_dataset_name", "Persisted Training Dataset"
        ) or "Persisted Training Dataset"
        set_ml_state(
            st.session_state.ml_dataset_name, trained_target, features,
            X_train, X_test, y_train, y_test, results, pred_table,
            trained_models, X, y, df
        )
        st.session_state.ml_restore_status = (
            "Saved training dataset found. Models were automatically retrained and restored."
        )
        return True
    except Exception as exc:
        st.session_state.ml_restore_status = (
            f"Automatic model recovery failed: {exc}"
        )
        print(f"GeoPolymer AI auto-recovery warning: {exc}")
        return False


def clear_ml_training_state():
    """Clear trained-model state when a genuinely new dataset is uploaded."""
    st.session_state.ml_models = {}
    st.session_state.ml_results = None
    st.session_state.ml_predictions = None
    st.session_state.ml_features = []
    st.session_state.ml_train_X = None
    st.session_state.ml_train_y = None
    st.session_state.ml_full_X = None
    st.session_state.ml_full_y = None
    st.session_state.ml_target = "28-Day Strength (MPa)"
    st.session_state.recommended_mix = None
    st.session_state.recommended_mix_label = "Primary Recommended Mix"
    st.session_state.optimization_results = None
    st.session_state.optimization_closest_mix = None
    st.session_state.optimization_higher_mix = None
    st.session_state.optimization_target_met = False
    st.session_state.optimization_target_met_closest = False
    st.session_state.optimization_target_met_higher = False
    st.session_state.cost_result = None
    st.session_state.co2_result = None
    st.session_state.quality_result = None
    st.session_state.smart_result = None


def seed_last_confirmed_project_state():
    """Restore the latest confirmed project outputs on a fresh install."""
    if STATE_FILE.exists():
        return

    # Latest confirmed mix used in the project.
    rec = pd.Series({
        "GGBS (kg/m³)": 440.16,
        "NaOH Molarity (M)": 14.92,
        "Na₂SiO₃/NaOH Ratio": 2.167,
        "Activator/Binder Ratio": 0.528,
        "Water/Binder Ratio": 0.329,
        "Fine Aggregate (kg/m³)": 700.0,
        "Coarse Aggregate (kg/m³)": 1100.0,
        "Superplasticizer (kg/m³)": 0.54,
        "Curing Temperature (°C)": 79.24,
        "Curing Duration (h)": 45.84,
        "Slump (mm)": 100.0,
        "Density (kg/m³)": 2350.0,
        "Predicted 28-Day Strength (MPa)": 40.81,
        "Strength Gap to Target (MPa)": 0.81,
        "Target Achieved": True,
        "Absolute Error vs Target (MPa)": 0.81
    })

    closest = rec.copy()
    closest["Predicted 28-Day Strength (MPa)"] = 40.00
    closest["Strength Gap to Target (MPa)"] = 0.00
    closest["Target Achieved"] = False
    closest["Absolute Error vs Target (MPa)"] = 0.00

    # The project had a confirmed SVM-best comparison table.
    st.session_state.ml_results = pd.DataFrame([
        {"Model": "SVM", "R²": 0.6564, "MAE (MPa)": 2.964, "RMSE (MPa)": 3.929, "Rank": 1},
        {"Model": "XGBoost", "R²": 0.6131, "MAE (MPa)": 3.127, "RMSE (MPa)": 4.170, "Rank": 2},
        {"Model": "Random Forest", "R²": 0.4717, "MAE (MPa)": 3.794, "RMSE (MPa)": 4.872, "Rank": 3},
        {"Model": "ANN", "R²": 0.1098, "MAE (MPa)": 4.888, "RMSE (MPa)": 6.324, "Rank": 4}
    ])
    st.session_state.ml_features = list(FEATURE_CANDIDATES)
    st.session_state.ml_target = "28-Day Strength (MPa)"
    st.session_state.ml_dataset_name = "Last Confirmed Project State"

    st.session_state.optimization_target = 40.0
    st.session_state.optimization_closest_mix = closest
    st.session_state.optimization_higher_mix = rec
    st.session_state.optimization_results = pd.DataFrame([closest, rec])
    st.session_state.recommended_mix = rec
    st.session_state.recommended_mix_label = "Higher-Strength Target-Achieving Mix"
    st.session_state.optimization_target_met = True
    st.session_state.optimization_target_met_closest = False
    st.session_state.optimization_target_met_higher = True

    # Keep the current dashboard inputs aligned with the confirmed recommendation.
    st.session_state.ggbs = 440.16
    st.session_state.activator_ratio = 0.528
    st.session_state.naoh_molarity = 14.92
    st.session_state.silicate_naoh_ratio = 2.167
    st.session_state.water_binder = 0.329
    st.session_state.fine_aggregate = 700.0
    st.session_state.coarse_aggregate = 1100.0
    st.session_state.superplasticizer = 0.54
    st.session_state.curing_temperature = 79.24
    st.session_state.curing_duration = 45.84
    st.session_state.slump = 100.0
    st.session_state.density = 2350.0

    st.session_state.cost_result = {
        "total": 14627.91,
        "breakdown": pd.DataFrame(),
        "mix_label": "Higher-Strength Target-Achieving Mix",
        "predicted_strength": 40.81
    }

    st.session_state.co2_result = {
        "total": 299.98,
        "breakdown": pd.DataFrame()
    }

    st.session_state.quality_result = {
        "strength_score": 100.0,
        "workability_score": 100.0,
        "cost_score": 95.7,
        "co2_score": 66.7,
        "overall": 92.5,
        "weights": {
            "Strength": 40.0,
            "Workability": 20.0,
            "Cost": 20.0,
            "CO₂": 20.0
        },
        "references": {
            "Target Strength (MPa)": 40.0,
            "Cost Reference (₹/m³)": 14000.0,
            "CO₂ Reference (kg CO₂e/m³)": 200.0
        }
    }

    st.session_state.crack_score = 88.4
    st.session_state.crack_area = 2.09
    st.session_state.crack_screening = "CRACK-LIKE"
    st.session_state.crack_severity = "High"
    st.session_state.crack_image_name = "Last Confirmed Crack Screening"

    st.session_state.smart_result = {
        "status": "VALIDATION REQUIRED",
        "message": (
            "Complete laboratory and engineering validation before "
            "treating the recommendation as final."
        ),
        "attention": [
            "Crack screening is High and requires visual/engineering assessment before acceptance."
        ],
        "actions": [
            "Repeat crack inspection using validated computer-vision data and perform engineering assessment.",
            "Perform laboratory compressive-strength and workability validation before treating the mix as final."
        ]
    }

    save_project_state()


# Load last saved project state, or seed the latest confirmed project state.
_state_loaded = load_project_state()
# V2 starts clean unless a previously saved V2 project state exists.
# This prevents hard-coded/demo results from being presented as fresh experimental results.
_model_restored = load_model_bundle()


def get_best_model_name():
    if st.session_state.get("ml_results") is not None:
        try:
            return str(st.session_state.ml_results.iloc[0]["Model"])
        except Exception:
            pass
    return "SVM" if st.session_state.get("ml_models") and "SVM" in st.session_state.ml_models else "Not trained"


def build_optimization_summary():
    rec = st.session_state.get("recommended_mix")
    closest = st.session_state.get("optimization_closest_mix")
    higher = st.session_state.get("optimization_higher_mix")
    target = float(st.session_state.get("optimization_target", 40.0))
    if rec is None and closest is None and higher is None:
        return None

    def strength(row):
        if row is None:
            return None
        try:
            return float(row["Predicted 28-Day Strength (MPa)"])
        except Exception:
            return None

    cs = strength(closest)
    hs = strength(higher)
    ps = strength(rec)
    target_met = bool(ps is not None and ps >= target - 1e-6)

    return {
        "target": target,
        "primary_strength": ps,
        "primary_label": st.session_state.get("recommended_mix_label", "Primary Recommended Mix"),
        "target_achieved": target_met,
        "closest_strength": cs,
        "closest_target_achieved": bool(cs is not None and cs >= target - 1e-6),
        "highest_strength": hs,
        "highest_target_achieved": bool(hs is not None and hs >= target - 1e-6),
    }


def _pdf_font_family():
    # Prefer Times New Roman when installed; otherwise use a Times-compatible serif.
    try:
        import matplotlib.font_manager as fm
        fm.findfont(
            fm.FontProperties(family="Times New Roman"),
            fallback_to_default=False
        )
        return "Times New Roman"
    except Exception:
        return "DejaVu Serif"


def _wrap_for_pdf(text, width=95):
    import textwrap
    return textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False) or [""]


def make_simple_pdf(title, sections, filename_hint):
    """
    Dependency-free PDF generator using matplotlib/PdfPages.
    Uses Times New Roman when available and falls back to a Times-compatible serif.
    sections = list of (heading, body_lines)
    """
    import io
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    family = _pdf_font_family()
    buffer = io.BytesIO()

    pages = []
    current_lines = [title, "", f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"]

    # Page-height-aware line budgeting.
    max_lines = 43

    def flush():
        nonlocal current_lines
        if current_lines:
            pages.append(current_lines)
        current_lines = []

    for heading, body in sections:
        block = [heading]
        for line in body:
            for wrapped in _wrap_for_pdf(line, 94):
                block.append(wrapped)
        block.append("")
        if len(current_lines) + len(block) > max_lines and len(current_lines) > 4:
            flush()
        current_lines.extend(block)
        # If a very large section still exceeds one page, split naturally.
        while len(current_lines) > max_lines:
            part = current_lines[:max_lines]
            pages.append(part)
            current_lines = [f"{heading} (continued)"] + current_lines[max_lines:]
    flush()

    with PdfPages(buffer) as pdf:
        for page_lines in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.09, 0.07, 0.82, 0.86])
            ax.axis("off")

            y = 0.98
            first = True
            for line in page_lines:
                if first:
                    ax.text(
                        0.5, y, line, ha="center", va="top",
                        fontsize=17, fontweight="bold", family=family
                    )
                    y -= 0.055
                    first = False
                elif line.startswith("Generated:"):
                    ax.text(
                        0.5, y, line, ha="center", va="top",
                        fontsize=9, family=family
                    )
                    y -= 0.038
                elif line and (
                    line[0].isdigit() and ". " in line[:4]
                    or line.isupper()
                    or line.startswith("📌")
                    or line.startswith("🎯")
                    or line.startswith("⚠️")
                ):
                    ax.text(
                        0.0, y, line, ha="left", va="top",
                        fontsize=12.5, fontweight="bold", family=family
                    )
                    y -= 0.043
                else:
                    ax.text(
                        0.0, y, line, ha="left", va="top",
                        fontsize=10.5, family=family
                    )
                    y -= 0.032

                if y < 0.035:
                    break

            ax.text(
                0.5, 0.015,
                f"GeoPolymer AI | {filename_hint}",
                ha="center", va="bottom",
                fontsize=8.5, family=family
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def build_standard_sections():
    best = None
    if st.session_state.get("ml_results") is not None:
        best = st.session_state.ml_results.iloc[0]

    opt = build_optimization_summary()
    rec = st.session_state.get("recommended_mix")
    cost = st.session_state.get("cost_result")
    co2 = st.session_state.get("co2_result")
    quality = st.session_state.get("quality_result")
    try:
        smart = build_smart_recommendation()
    except Exception:
        smart = st.session_state.get("smart_result")

    if smart is not None:
        st.session_state.smart_result = smart

    ml_lines = []
    if best is not None:
        ml_lines.append(f"Best ML Model: {best['Model']}")
        ml_lines.append(f"R²: {best['R²']:.4f}")
        ml_lines.append(f"MAE: {best['MAE (MPa)']:.3f} MPa")
        ml_lines.append(f"RMSE: {best['RMSE (MPa)']:.3f} MPa")
    else:
        ml_lines.append("No ML training result available.")

    opt_lines = []
    if opt:
        opt_lines.extend([
            f"Target strength: {opt['target']:.2f} MPa",
            f"Primary recommendation: {opt['primary_label']}",
            f"Primary predicted strength: {opt['primary_strength']:.2f} MPa",
            f"Target achieved: {'YES' if opt['target_achieved'] else 'NO'}",
            f"Closest candidate: {opt['closest_strength']:.2f} MPa",
            f"Closest target achieved: {'YES' if opt['closest_target_achieved'] else 'NO'}",
            f"Highest candidate: {opt['highest_strength']:.2f} MPa",
            f"Highest target achieved: {'YES' if opt['highest_target_achieved'] else 'NO'}",
        ])
    else:
        opt_lines.append("Optimization not run.")

    rec_lines = []
    if rec is not None:
        rec_lines = [
            f"Best ML Model: {get_best_model_name()}",
            f"GGBS: {float(rec['GGBS (kg/m³)']):.2f} kg/m³",
            f"NaOH molarity: {float(rec['NaOH Molarity (M)']):.2f} M",
            f"Silicate/NaOH ratio: {float(rec['Na₂SiO₃/NaOH Ratio']):.3f}",
            f"Activator/Binder ratio: {float(rec['Activator/Binder Ratio']):.3f}",
            f"Water/Binder ratio: {float(rec['Water/Binder Ratio']):.3f}",
            f"Fine aggregate: {float(rec['Fine Aggregate (kg/m³)']):.2f} kg/m³",
            f"Coarse aggregate: {float(rec['Coarse Aggregate (kg/m³)']):.2f} kg/m³",
            f"Superplasticizer: {float(rec['Superplasticizer (kg/m³)']):.2f} kg/m³",
            f"Curing temperature: {float(rec['Curing Temperature (°C)']):.2f} °C",
            f"Curing duration: {float(rec['Curing Duration (h)']):.2f} h",
            f"Predicted 28-day strength: {float(rec['Predicted 28-Day Strength (MPa)']):.2f} MPa",
        ]
    else:
        rec_lines.append("No recommended mix available.")

    cost_lines = [
        f"Estimated material cost: ₹{float(cost['total']):,.2f} / m³"
        if cost else "Cost analysis not calculated."
    ]
    co2_lines = [
        f"Indicative material CO₂: {float(co2['total']):.2f} kg CO₂e / m³"
        if co2 else "CO₂ analysis not calculated."
    ]
    quality_lines = [
        f"Overall quality score: {float(quality['overall']):.1f} / 100"
        if quality else "Quality score not calculated."
    ]
    smart_lines = [
        f"System recommendation: {smart['status']}",
        f"Decision message: {smart['message']}"
    ] if smart else ["Smart recommendation not available."]

    crack_lines = []
    if st.session_state.get("crack_score") is not None:
        crack_lines = [
            f"Detection score: {float(st.session_state.crack_score):.1f}/100",
            f"Estimated crack area: {float(st.session_state.crack_area):.2f}%",
            f"Screening: {st.session_state.get('crack_screening')}",
            f"Severity: {st.session_state.get('crack_severity')}",
            f"Image: {st.session_state.get('crack_image_name', '')}",
        ]
    else:
        crack_lines = ["No crack image has been analysed in this session."]

    return [
        ("1. Executive Summary", [
            "GeoPolymer AI is an integrated AI/ML-based decision-support platform for 100% GGBS geopolymer concrete.",
            f"Best ML model: {get_best_model_name()}",
            f"Optimization target: {st.session_state.get('optimization_target', 40.0):.2f} MPa",
            f"Primary predicted strength: {float(rec['Predicted 28-Day Strength (MPa)']):.2f} MPa" if rec is not None else "Primary predicted strength: not available",
            "Final adoption requires laboratory validation, verified inputs and engineering review."
        ]),
        ("2. Mix Design", [
            "Precursor system: 100% GGBS",
            f"GGBS: {st.session_state.get('ggbs', 400.0):.2f} kg/m³",
            f"Activator/GGBS ratio: {st.session_state.get('activator_ratio', 0.40):.3f}",
            f"NaOH molarity: {st.session_state.get('naoh_molarity', 10.0):.2f} M",
            f"Silicate/NaOH ratio: {st.session_state.get('silicate_naoh_ratio', 2.0):.3f}",
            f"Water/Binder ratio: {st.session_state.get('water_binder', 0.35):.3f}",
            f"Fine aggregate: {st.session_state.get('fine_aggregate', 700.0):.2f} kg/m³",
            f"Coarse aggregate: {st.session_state.get('coarse_aggregate', 1100.0):.2f} kg/m³",
            f"Superplasticizer: {st.session_state.get('superplasticizer', 1.0):.2f} kg/m³",
        ]),
        ("3. AI / ML Model Performance", ml_lines),
        ("4. Optimization", opt_lines),
        ("5. Primary Recommended Mix", rec_lines),
        ("6. Cost Analysis", cost_lines),
        ("7. CO₂ Analysis", co2_lines),
        ("8. Multi-Criteria Performance Score", quality_lines),
        ("9. Smart Recommendation", smart_lines),
        ("10. Crack Screening", crack_lines),
        ("11. Final Validation Note", [
            "Demo/synthetic ML data are for software pipeline testing and must be replaced by verified laboratory records.",
            "CO₂ values are indicative until documented project-specific environmental data are entered.",
            "Crack screening is preliminary image analysis and is not a structural safety assessment.",
            "Laboratory compressive-strength, workability and other required engineering validations remain necessary."
        ])
    ]


def build_descriptive_sections():
    sections = build_standard_sections()
    # Append descriptive narratives to the corresponding sections.
    expanded = []
    for heading, body in sections:
        if heading == "3. AI / ML Model Performance":
            body += [
                "Problem: Model performance can change when the dataset is small, synthetic or not representative of real concrete.",
                "Possible factors: dataset size, feature distribution, train/test split sensitivity, measurement noise and differences between demo and laboratory data.",
                "Solution: replace demo data with verified laboratory data, expand the dataset, use repeated cross-validation and confirm performance on a laboratory hold-out set."
            ]
        elif heading == "4. Optimization":
            body += [
                "Problem: a displayed value may round to the target even when its underlying prediction is slightly below it.",
                "Factor: floating-point precision and the unrounded model prediction.",
                "Solution: use the underlying prediction for target-achievement logic and clearly show both closest and highest candidates."
            ]
        elif heading == "6. Cost Analysis":
            body += [
                "Problem: estimated cost is sensitive to market and supplier prices.",
                "Possible factors: supplier, location, purchase volume, material grade, activator chemistry and changes in recommended proportions.",
                "Solution: use dated and verified local/supplier rates and recalculate whenever the mix changes."
            ]
        elif heading == "7. CO₂ Analysis":
            body += [
                "Problem: the current CO₂ value is an indicative material-emissions estimate, not a complete life-cycle assessment.",
                "Possible factors: emission-factor source, system boundary, transportation, processing energy and material-specific inventories.",
                "Solution: replace illustrative factors with documented supplier EPD/LCA or environmental inventory data and state the system boundary."
            ]
        elif heading == "8. Multi-Criteria Performance Score":
            body += [
                "Problem: a high composite score can hide a weaker individual component.",
                "Possible factors: selected weights, reference values and uncertainty in model inputs.",
                "Solution: report every component score, justify the weighting method and replace prototype reference values with project-specific benchmarks."
            ]
        elif heading == "9. Smart Recommendation":
            body += [
                "Problem: the recommendation is model-based and cannot replace physical validation.",
                "Factor: the primary strength prediction is based on the current trained model and current input assumptions.",
                "Solution: perform laboratory compressive-strength and workability validation before treating the mix as final."
            ]
        elif heading == "10. Crack Screening":
            body += [
                "Problem: image-based screening can be affected by shadows, stains, texture, lighting, camera angle and thresholds.",
                "Possible factors: image quality, surface texture, aggregate contrast and screening rules.",
                "Solution: build a labelled crack-image dataset, validate a computer-vision model against expert labels and use engineering inspection for structural assessment."
            ]
        expanded.append((heading, body))
    return expanded


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_activator():
    total = (
        st.session_state.ggbs
        * st.session_state.activator_ratio
    )
    naoh = total / (
        1.0 + st.session_state.silicate_naoh_ratio
    )
    silicate = total - naoh
    return total, naoh, silicate


def train_model_bundle(df):
    available_targets = [
        c for c in TARGET_OPTIONS
        if c in df.columns
    ]

    if not available_targets:
        raise ValueError(
            "No strength target columns were found."
        )

    target = (
        st.session_state.ml_target
        if st.session_state.ml_target in available_targets
        else available_targets[0]
    )

    features = [
        c for c in FEATURE_CANDIDATES
        if c in df.columns
    ]

    if len(features) < 3:
        raise ValueError(
            "Fewer than 3 usable ML features were found."
        )

    clean = df[features + [target]].copy()

    for col in features:
        clean[col] = pd.to_numeric(
            clean[col],
            errors="coerce"
        )

    clean[target] = pd.to_numeric(
        clean[target],
        errors="coerce"
    )

    clean = clean.dropna(
        subset=[target]
    )

    if len(clean) < 20:
        raise ValueError(
            "At least 20 valid rows are recommended for demo training."
        )

    X = clean[features]
    y = clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    trained_models = {}
    predictions = {}
    metrics = []
    cv_splits = min(5, len(X_train))
    if cv_splits < 3:
        raise ValueError("At least 3 training folds are required for cross-validation.")
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    def add_cv_metrics(model, row):
        cv_out = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row["CV R² Mean"] = float(np.mean(cv_out["test_r2"]))
        row["CV R² Std"] = float(np.std(cv_out["test_r2"]))
        row["CV MAE (MPa)"] = float(-np.mean(cv_out["test_mae"]))
        row["CV RMSE (MPa)"] = float(-np.mean(cv_out["test_rmse"]))
        return row

    if XGBOOST_AVAILABLE:
        xgb = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=42
                )
            )
        ])

        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)

        trained_models["XGBoost"] = xgb
        predictions["XGBoost"] = pred

        metrics.append(add_cv_metrics(xgb, {
            "Model": "XGBoost",
            "R²": r2_score(y_test, pred),
            "MAE (MPa)": mean_absolute_error(y_test, pred),
            "RMSE (MPa)": np.sqrt(
                mean_squared_error(y_test, pred)
            )
        }))

    rf = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    trained_models["Random Forest"] = rf
    predictions["Random Forest"] = rf_pred

    metrics.append(add_cv_metrics(rf, {
        "Model": "Random Forest",
        "R²": r2_score(y_test, rf_pred),
        "MAE (MPa)": mean_absolute_error(y_test, rf_pred),
        "RMSE (MPa)": np.sqrt(
            mean_squared_error(y_test, rf_pred)
        )
    }))

    svm = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVR(
                kernel="rbf",
                C=100,
                epsilon=0.1
            )
        )
    ])

    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    trained_models["SVM"] = svm
    predictions["SVM"] = svm_pred

    metrics.append(add_cv_metrics(svm, {
        "Model": "SVM",
        "R²": r2_score(y_test, svm_pred),
        "MAE (MPa)": mean_absolute_error(y_test, svm_pred),
        "RMSE (MPa)": np.sqrt(
            mean_squared_error(y_test, svm_pred)
        )
    }))

    ann = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=1500,
                early_stopping=True,
                random_state=42
            )
        )
    ])

    ann.fit(X_train, y_train)
    ann_pred = ann.predict(X_test)

    trained_models["ANN"] = ann
    predictions["ANN"] = ann_pred

    metrics.append(add_cv_metrics(ann, {
        "Model": "ANN",
        "R²": r2_score(y_test, ann_pred),
        "MAE (MPa)": mean_absolute_error(y_test, ann_pred),
        "RMSE (MPa)": np.sqrt(
            mean_squared_error(y_test, ann_pred)
        )
    }))

    results = pd.DataFrame(metrics).sort_values(
        "R²",
        ascending=False
    ).reset_index(drop=True)

    results["Rank"] = np.arange(
        1,
        len(results) + 1
    )
    results["Reliability"] = np.select(
        [results["CV R² Mean"] >= 0.80, results["CV R² Mean"] >= 0.60, results["CV R² Mean"] >= 0.40],
        ["Strong", "Moderate", "Limited"],
        default="Weak"
    )

    pred_table = pd.DataFrame(
        predictions,
        index=y_test.index
    )
    pred_table["Actual"] = y_test.values
    pred_table["Mix Index"] = y_test.index
    # Store model error estimates for transparent prediction confidence reporting.
    cv_rmse_map = dict(zip(results["Model"], results["CV RMSE (MPa)"]))
    for model_name in predictions:
        pred_table[f"{model_name} ± CV-RMSE (MPa)"] = cv_rmse_map.get(model_name, np.nan)

    return (
        target,
        features,
        X_train,
        X_test,
        y_train,
        y_test,
        results,
        pred_table,
        trained_models,
        X,
        y
    )


def set_ml_state(
    dataset_name,
    target,
    features,
    X_train,
    X_test,
    y_train,
    y_test,
    results,
    pred_table,
    trained_models,
    X,
    y,
    df
):
    st.session_state.ml_dataset_name = dataset_name
    st.session_state.ml_target = target
    st.session_state.ml_features = features
    st.session_state.ml_train_X = X_train
    st.session_state.ml_train_y = y_train
    st.session_state.ml_full_X = X
    st.session_state.ml_full_y = y
    st.session_state.ml_results = results
    st.session_state.ml_predictions = pred_table
    st.session_state.ml_models = trained_models
    st.session_state.ml_dataset = df

    # A new training run becomes the authoritative saved model bundle.
    save_model_bundle()
    save_project_state()


# Final fallback after all ML helper functions are defined.
# This is intentionally after train_model_bundle() so a saved CSV can be
# used to retrain automatically if the serialized model cannot be loaded.
if not st.session_state.get("ml_models"):
    auto_recover_trained_models()


def get_recommended_mix_strength():
    if st.session_state.recommended_mix is None:
        return None
    return float(
        st.session_state.recommended_mix[
            "Predicted 28-Day Strength (MPa)"
        ]
    )


def calculate_cost_for_mix(mix):
    total_activator = (
        float(mix["GGBS (kg/m³)"])
        * float(mix["Activator/Binder Ratio"])
    )
    naoh = total_activator / (
        1.0 + float(mix["Na₂SiO₃/NaOH Ratio"])
    )
    silicate = total_activator - naoh

    rows = [
        (
            "GGBS",
            float(mix["GGBS (kg/m³)"]),
            st.session_state.cost_rates["GGBS"]
        ),
        (
            "NaOH",
            naoh,
            st.session_state.cost_rates["NaOH"]
        ),
        (
            "Sodium Silicate",
            silicate,
            st.session_state.cost_rates["Sodium Silicate"]
        ),
        (
            "Fine Aggregate",
            float(mix["Fine Aggregate (kg/m³)"]),
            st.session_state.cost_rates["Fine Aggregate"]
        ),
        (
            "Coarse Aggregate",
            float(mix["Coarse Aggregate (kg/m³)"]),
            st.session_state.cost_rates["Coarse Aggregate"]
        ),
        (
            "Superplasticizer",
            float(mix["Superplasticizer (kg/m³)"]),
            st.session_state.cost_rates["Superplasticizer"]
        )
    ]

    cost_df = pd.DataFrame(
        rows,
        columns=[
            "Material",
            "Quantity (kg/m³)",
            "Rate (₹/kg)"
        ]
    )
    cost_df["Cost (₹/m³)"] = (
        cost_df["Quantity (kg/m³)"]
        * cost_df["Rate (₹/kg)"]
    )

    return cost_df, float(
        cost_df["Cost (₹/m³)"].sum()
    )


def calculate_co2_for_mix(mix):
    total_activator = (
        float(mix["GGBS (kg/m³)"])
        * float(mix["Activator/Binder Ratio"])
    )
    naoh = total_activator / (
        1.0 + float(mix["Na₂SiO₃/NaOH Ratio"])
    )
    silicate = total_activator - naoh

    rows = [
        (
            "GGBS",
            float(mix["GGBS (kg/m³)"]),
            st.session_state.co2_factors["GGBS"]
        ),
        (
            "NaOH",
            naoh,
            st.session_state.co2_factors["NaOH"]
        ),
        (
            "Sodium Silicate",
            silicate,
            st.session_state.co2_factors["Sodium Silicate"]
        ),
        (
            "Fine Aggregate",
            float(mix["Fine Aggregate (kg/m³)"]),
            st.session_state.co2_factors["Fine Aggregate"]
        ),
        (
            "Coarse Aggregate",
            float(mix["Coarse Aggregate (kg/m³)"]),
            st.session_state.co2_factors["Coarse Aggregate"]
        ),
        (
            "Superplasticizer",
            float(mix["Superplasticizer (kg/m³)"]),
            st.session_state.co2_factors["Superplasticizer"]
        )
    ]

    co2_df = pd.DataFrame(
        rows,
        columns=[
            "Material",
            "Quantity (kg/m³)",
            "Emission Factor (kg CO₂e/kg)"
        ]
    )

    co2_df["CO₂e (kg/m³)"] = (
        co2_df["Quantity (kg/m³)"]
        * co2_df["Emission Factor (kg CO₂e/kg)"]
    )

    return co2_df, float(
        co2_df["CO₂e (kg/m³)"].sum()
    )


def calculate_quality():
    strength = get_recommended_mix_strength()

    if strength is None:
        return None

    target = float(
        st.session_state.optimization_target
    )

    cost_value = None
    co2_value = None

    if st.session_state.cost_result is not None:
        cost_value = float(
            st.session_state.cost_result["total"]
        )

    if st.session_state.co2_result is not None:
        co2_value = float(
            st.session_state.co2_result["total"]
        )

    if cost_value is None or co2_value is None:
        return None

    strength_score = min(
        100.0,
        max(0.0, (strength / target) * 100.0)
    )

    # Transparent prototype workability score.
    # Full workability validation still requires lab testing.
    workability_target_low = 75.0
    workability_target_high = 150.0

    slump = float(
        st.session_state.recommended_mix["Slump (mm)"]
    )

    if (
        workability_target_low
        <= slump
        <= workability_target_high
    ):
        workability_score = 100.0
    else:
        distance = min(
            abs(slump - workability_target_low),
            abs(slump - workability_target_high)
        )
        workability_score = max(
            0.0,
            100.0 - distance * 2.0
        )

    cost_reference = 14000.0
    cost_score = min(
        100.0,
        max(0.0, (cost_reference / cost_value) * 100.0)
    )

    co2_reference = 200.0
    co2_score = min(
        100.0,
        max(0.0, (co2_reference / co2_value) * 100.0)
    )

    score = (
        0.40 * strength_score
        + 0.20 * workability_score
        + 0.20 * cost_score
        + 0.20 * co2_score
    )

    return {
        "score_name": "Multi-Criteria Performance Score (MCPS)",
        "strength_score": strength_score,
        "workability_score": workability_score,
        "cost_score": cost_score,
        "co2_score": co2_score,
        "overall": min(100.0, max(0.0, score)),
        "weights": {
            "Strength": 40.0,
            "Workability": 20.0,
            "Cost": 20.0,
            "CO₂": 20.0
        },
        "references": {
            "Target Strength (MPa)": target,
            "Cost Reference (₹/m³)": cost_reference,
            "CO₂ Reference (kg CO₂e/m³)": co2_reference
        }
    }


def build_smart_recommendation():
    strength = get_recommended_mix_strength()

    if strength is None:
        return {
            "status": "VALIDATION REQUIRED",
            "message": (
                "Run Mix Optimization after training the ML model."
            ),
            "attention": [
                "No recommended mix is available."
            ],
            "actions": [
                "Train the ML model.",
                "Run Mix Optimization."
            ]
        }

    target = float(
        st.session_state.optimization_target
    )

    attention = []
    actions = []

    # A recommendation cannot be treated as robust when cross-validation
    # indicates weak predictive performance.
    try:
        best_row = st.session_state.ml_results.iloc[0]
        cv_r2 = float(best_row.get("CV R² Mean", np.nan))
        if np.isfinite(cv_r2) and cv_r2 < 0.60:
            attention.append(f"Best model cross-validation R² is only {cv_r2:.3f}; predictive reliability is limited.")
            actions.append("Collect more representative laboratory data and retrain before relying on the optimization result.")
    except Exception:
        attention.append("Cross-validation reliability information is unavailable.")
        actions.append("Train the model with the improved cross-validation workflow before final decision-making.")

    if strength < target:
        attention.append(
            f"Predicted 28-day strength ({strength:.2f} MPa) "
            f"is {target - strength:.2f} MPa below the target."
        )
        actions.append(
            "Adjust/optimize the mix until the predicted strength "
            "reaches the target, then verify experimentally."
        )

    crack_high = (
        st.session_state.crack_severity == "High"
    )

    if crack_high:
        attention.append(
            "Crack screening is High and requires visual/engineering "
            "assessment before acceptance."
        )
        actions.append(
            "Repeat crack inspection using validated computer-vision "
            "data and perform engineering assessment."
        )

    if (
        st.session_state.cost_result is None
        or st.session_state.co2_result is None
    ):
        actions.append(
            "Complete cost and CO₂ analysis for the recommended mix."
        )

    if st.session_state.quality_result is None:
        actions.append(
            "Calculate the Multi-Criteria Performance Score after cost and CO₂ analysis."
        )

    actions.append(
        "Perform laboratory compressive-strength and workability "
        "validation before treating the mix as final."
    )

    if not attention and not crack_high:
        status = "PRELIMINARILY RECOMMENDED"
        message = (
            "The current model-based checks support preliminary "
            "recommendation, subject to laboratory validation."
        )
    else:
        status = "VALIDATION REQUIRED"
        message = (
            "Complete laboratory and engineering validation before "
            "treating the recommendation as final."
        )

    return {
        "status": status,
        "message": message,
        "attention": attention,
        "evidence_level": "Model-based screening; laboratory validation required",
        "uncertainty_note": "Prediction uncertainty is represented by cross-validation error, not a certified confidence interval.",
        "actions": list(dict.fromkeys(actions))
    }


def analyze_crack_image(image):
    """
    Preliminary image screening only.
    Uses grayscale contrast/thresholding; not a trained structural crack model.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError(
            "Pillow is not available in this Python environment."
        )

    img = image.convert("L")
    arr = np.array(img).astype(np.float32)

    # Local contrast proxy using a small average pool via pandas rolling.
    gray = pd.DataFrame(arr)
    smooth = (
        gray.rolling(7, center=True, min_periods=1)
        .mean()
        .rolling(7, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    dark_residual = np.clip(
        smooth - arr,
        0,
        None
    )

    # Threshold tuned conservatively for a preliminary crack-like screen.
    threshold = np.percentile(
        dark_residual,
        98.5
    )

    mask = dark_residual >= threshold

    area = float(mask.mean() * 100.0)

    # Score combines the detected area and contrast strength.
    contrast_score = min(
        100.0,
        float(
            np.mean(
                np.clip(
                    dark_residual / max(
                        1.0,
                        np.percentile(
                            dark_residual,
                            99.5
                        )
                    ),
                    0,
                    1
                )
            ) * 250.0
        )
    )

    detection_score = min(
        100.0,
        40.0
        + area * 8.0
        + contrast_score * 0.7
    )

    if area < 0.50:
        severity = "Low"
        screening = "LOW CRACK-LIKE"
    elif area < 1.50:
        severity = "Moderate"
        screening = "CRACK-LIKE"
    else:
        severity = "High"
        screening = "CRACK-LIKE"

    return (
        round(detection_score, 1),
        round(area, 2),
        screening,
        severity,
        mask
    )


# =========================================================
# NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧱 Mix Design",
        "🔬 Material Parameters",
        "⚗️ Activator",
        "📊 Mix Parameters",
        "📈 Strength Prediction",
        "🧪 Lab Data",
        "🤖 AI / XGBoost",
        "🎯 Mix Optimization",
        "💰 Cost Analysis",
        "🌱 CO₂ Analysis",
        "⭐ Performance Score",
        "🧠 Smart Recommendation",
        "📷 Crack Detection",
        "📊 Results & Analysis",
        "📄 PDF Report"
    ],
    key="main_navigation"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🏗️ GeoPolymer AI")
    st.header(
        "Integrated Geopolymer Concrete Mix Design & AI Decision Support"
    )

    st.info(
        "100% GGBS-based platform. Demo/synthetic ML data are for "
        "software and pipeline testing only; final recommendations "
        "require laboratory validation."
    )

    best_model = get_best_model_name()
    strength = get_recommended_mix_strength()
    quality = (
        st.session_state.quality_result["overall"]
        if st.session_state.quality_result
        else None
    )
    cost = (
        st.session_state.cost_result["total"]
        if st.session_state.cost_result
        else None
    )
    co2 = (
        st.session_state.co2_result["total"]
        if st.session_state.co2_result
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precursor",
        "100% GGBS"
    )

    c2.metric(
        "Best Model",
        best_model or "Not trained"
    )

    c3.metric(
        "Recommended Strength",
        f"{strength:.2f} MPa" if strength else "Not available"
    )

    c4.metric(
        "MCPS",
        f"{quality:.1f}/100" if quality is not None else "Pending"
    )

    st.divider()

    st.header("💰🌱 Decision-Support Indicators")

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Material Cost",
        f"₹{cost:,.2f}/m³" if cost is not None else "Pending"
    )

    d2.metric(
        "Indicative CO₂",
        f"{co2:.2f} kg/m³" if co2 is not None else "Pending"
    )

    if st.session_state.smart_result:
        d3.metric(
            "Recommendation",
            st.session_state.smart_result["status"]
        )
    else:
        d3.metric(
            "Recommendation",
            "Pending"
        )

    if st.session_state.recommended_mix is not None:

        st.header("🏆 Recommended Mix")

        rec = st.session_state.recommended_mix.copy()

        pretty = pd.DataFrame({
            "Parameter": [
                "Best ML Model",
                "GGBS",
                "NaOH Molarity",
                "Silicate/NaOH Ratio",
                "Activator/Binder Ratio",
                "Water/Binder Ratio",
                "Fine Aggregate",
                "Coarse Aggregate",
                "Superplasticizer",
                "Curing Temperature",
                "Curing Duration",
                "Slump",
                "Density",
                "Predicted 28-Day Strength"
            ],
            "Value": [
                best_model or "N/A",
                f"{rec['GGBS (kg/m³)']:.2f} kg/m³",
                f"{rec['NaOH Molarity (M)']:.2f} M",
                f"{rec['Na₂SiO₃/NaOH Ratio']:.3f}",
                f"{rec['Activator/Binder Ratio']:.3f}",
                f"{rec['Water/Binder Ratio']:.3f}",
                f"{rec['Fine Aggregate (kg/m³)']:.2f} kg/m³",
                f"{rec['Coarse Aggregate (kg/m³)']:.2f} kg/m³",
                f"{rec['Superplasticizer (kg/m³)']:.2f} kg/m³",
                f"{rec['Curing Temperature (°C)']:.2f} °C",
                f"{rec['Curing Duration (h)']:.1f} h",
                f"{rec['Slump (mm)']:.1f} mm",
                f"{rec['Density (kg/m³)']:.1f} kg/m³",
                f"{rec['Predicted 28-Day Strength (MPa)']:.2f} MPa"
            ]
        })

        st.dataframe(
            pretty,
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.crack_score is not None:

        st.header("📷 Latest Crack Screening")

        crack_df = pd.DataFrame({
            "Parameter": [
                "Detection Score",
                "Estimated Crack Area",
                "Screening",
                "Severity"
            ],
            "Value": [
                f"{st.session_state.crack_score:.1f}/100",
                f"{st.session_state.crack_area:.2f}%",
                st.session_state.crack_screening,
                st.session_state.crack_severity
            ]
        })

        st.dataframe(
            crack_df,
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.smart_result:

        sr = st.session_state.smart_result

        st.header("🧠 Smart Recommendation")

        if sr["status"] == "PRELIMINARILY RECOMMENDED":
            st.success(
                f"{sr['status']} — {sr['message']}"
            )
        else:
            st.warning(
                f"{sr['status']} — {sr['message']}"
            )


# =========================================================
# MIX DESIGN
# =========================================================

elif page == "🧱 Mix Design":

    st.title("🧱 Geopolymer Concrete Mix Design")
    st.success("Precursor system: 100% GGBS")

    st.session_state.ggbs = st.number_input(
        "GGBS / Binder (kg/m³)",
        min_value=100.0,
        max_value=800.0,
        value=float(st.session_state.ggbs),
        step=10.0
    )

    st.metric(
        "Precursor Percentage",
        "100% GGBS"
    )

    st.session_state.activator_ratio = st.number_input(
        "Activator / GGBS Ratio",
        min_value=0.10,
        max_value=1.00,
        value=float(st.session_state.activator_ratio),
        step=0.01
    )

    st.session_state.water_binder = st.number_input(
        "Water / Binder Ratio",
        min_value=0.10,
        max_value=0.80,
        value=float(st.session_state.water_binder),
        step=0.01
    )

    a, b = st.columns(2)

    with a:
        st.session_state.fine_aggregate = st.number_input(
            "Fine Aggregate (kg/m³)",
            min_value=400.0,
            max_value=1200.0,
            value=float(st.session_state.fine_aggregate),
            step=10.0
        )

    with b:
        st.session_state.coarse_aggregate = st.number_input(
            "Coarse Aggregate (kg/m³)",
            min_value=600.0,
            max_value=1600.0,
            value=float(st.session_state.coarse_aggregate),
            step=10.0
        )

    total_activator, naoh_part, silicate_part = (
        calculate_activator()
    )

    st.header("📋 Current Mix Composition")

    mix_table = pd.DataFrame({
        "Material": [
            "GGBS",
            "Total Alkaline Activator",
            "NaOH Portion",
            "Sodium Silicate",
            "Water",
            "Fine Aggregate",
            "Coarse Aggregate"
        ],
        "Quantity (kg/m³)": [
            st.session_state.ggbs,
            total_activator,
            naoh_part,
            silicate_part,
            st.session_state.ggbs
            * st.session_state.water_binder,
            st.session_state.fine_aggregate,
            st.session_state.coarse_aggregate
        ]
    })

    st.dataframe(
        mix_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MATERIAL PARAMETERS
# =========================================================

elif page == "🔬 Material Parameters":

    st.title("🔬 Material Parameters")

    sg = st.number_input(
        "GGBS Specific Gravity",
        min_value=1.0,
        max_value=5.0,
        value=2.90,
        step=0.01
    )

    fineness = st.number_input(
        "GGBS Blaine Fineness (m²/kg)",
        min_value=100.0,
        max_value=1000.0,
        value=400.0,
        step=10.0
    )

    material_df = pd.DataFrame({
        "Parameter": [
            "Material",
            "Percentage",
            "Specific Gravity",
            "Blaine Fineness"
        ],
        "Value": [
            "GGBS",
            "100%",
            f"{sg:.2f}",
            f"{fineness:.0f} m²/kg"
        ]
    })

    st.dataframe(
        material_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ACTIVATOR
# =========================================================

elif page == "⚗️ Activator":

    st.title("⚗️ Alkaline Activator")

    st.session_state.naoh_molarity = st.number_input(
        "NaOH Molarity (M)",
        min_value=1.0,
        max_value=20.0,
        value=float(st.session_state.naoh_molarity),
        step=0.5
    )

    st.session_state.silicate_naoh_ratio = st.number_input(
        "Silicate / NaOH Ratio",
        min_value=0.5,
        max_value=4.0,
        value=float(st.session_state.silicate_naoh_ratio),
        step=0.1
    )

    purity = st.number_input(
        "NaOH Purity (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0,
        step=0.5
    )

    total_activator, naoh_part, silicate_part = (
        calculate_activator()
    )

    theoretical_naoh = (
        st.session_state.naoh_molarity * 40.0
    )
    adjusted_naoh = theoretical_naoh / (
        purity / 100.0
    )

    activator_df = pd.DataFrame({
        "Parameter": [
            "Total Activator",
            "NaOH Portion",
            "Sodium Silicate",
            "NaOH Molarity",
            "NaOH Purity",
            "Theoretical NaOH Basis",
            "Purity-adjusted Basis"
        ],
        "Value": [
            f"{total_activator:.2f} kg/m³",
            f"{naoh_part:.2f} kg/m³",
            f"{silicate_part:.2f} kg/m³",
            f"{st.session_state.naoh_molarity:.2f} M",
            f"{purity:.1f}%",
            f"{theoretical_naoh:.2f} g/L",
            f"{adjusted_naoh:.2f} g/L"
        ]
    })

    st.dataframe(
        activator_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MIX PARAMETERS
# =========================================================

elif page == "📊 Mix Parameters":

    st.title("📊 Mix Parameters")

    params = pd.DataFrame({
        "Parameter": [
            "Precursor",
            "Precursor %",
            "GGBS",
            "Activator / GGBS",
            "NaOH Molarity",
            "Silicate / NaOH",
            "Water / Binder",
            "Fine Aggregate",
            "Coarse Aggregate"
        ],
        "Value": [
            "GGBS",
            "100%",
            f"{st.session_state.ggbs:.2f} kg/m³",
            f"{st.session_state.activator_ratio:.2f}",
            f"{st.session_state.naoh_molarity:.2f} M",
            f"{st.session_state.silicate_naoh_ratio:.2f}",
            f"{st.session_state.water_binder:.2f}",
            f"{st.session_state.fine_aggregate:.2f} kg/m³",
            f"{st.session_state.coarse_aggregate:.2f} kg/m³"
        ]
    })

    st.dataframe(
        params,
        use_container_width=True,
        hide_index=True
    )

    if st.session_state.recommended_mix is not None:
        st.info(
            "A recommended optimized mix is stored. "
            "Open 🎯 Mix Optimization or 🏠 Dashboard to review it."
        )


# =========================================================
# STRENGTH PREDICTION
# =========================================================

elif page == "📈 Strength Prediction":

    st.title("📈 Strength Prediction")

    st.header("🧮 New Mix Inputs")

    p1, p2, p3 = st.columns(3)

    with p1:
        pred_ggbs = st.number_input(
            "GGBS (kg/m³)",
            min_value=100.0,
            max_value=800.0,
            value=float(st.session_state.ggbs),
            step=10.0
        )

        pred_naoh = st.number_input(
            "NaOH Molarity (M)",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.naoh_molarity),
            step=0.5
        )

        pred_sio = st.number_input(
            "Silicate / NaOH Ratio",
            min_value=0.5,
            max_value=4.0,
            value=float(st.session_state.silicate_naoh_ratio),
            step=0.1
        )

        pred_ab = st.number_input(
            "Activator / GGBS Ratio",
            min_value=0.10,
            max_value=1.00,
            value=float(st.session_state.activator_ratio),
            step=0.01
        )

    with p2:
        pred_wb = st.number_input(
            "Water / Binder Ratio",
            min_value=0.10,
            max_value=0.80,
            value=float(st.session_state.water_binder),
            step=0.01
        )

        pred_fine = st.number_input(
            "Fine Aggregate (kg/m³)",
            min_value=400.0,
            max_value=1200.0,
            value=float(st.session_state.fine_aggregate),
            step=10.0
        )

        pred_coarse = st.number_input(
            "Coarse Aggregate (kg/m³)",
            min_value=600.0,
            max_value=1600.0,
            value=float(st.session_state.coarse_aggregate),
            step=10.0
        )

        pred_sp = st.number_input(
            "Superplasticizer (kg/m³)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.superplasticizer),
            step=0.1
        )

    with p3:
        pred_temp = st.number_input(
            "Curing Temperature (°C)",
            min_value=10.0,
            max_value=100.0,
            value=float(st.session_state.curing_temperature),
            step=1.0
        )

        pred_duration = st.number_input(
            "Curing Duration (h)",
            min_value=1.0,
            max_value=168.0,
            value=float(st.session_state.curing_duration),
            step=1.0
        )

        pred_slump = st.number_input(
            "Slump (mm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state.slump),
            step=1.0
        )

        pred_density = st.number_input(
            "Density (kg/m³)",
            min_value=1800.0,
            max_value=2600.0,
            value=float(st.session_state.density),
            step=10.0
        )

    prediction_age = st.selectbox(
        "Prediction Target Age",
        [7, 28, 56],
        index=1
    )

    age_target = {
        7: "7-Day Strength (MPa)",
        28: "28-Day Strength (MPa)",
        56: "56-Day Strength (MPa)"
    }[prediction_age]

    if (
        st.session_state.ml_models
        and age_target == st.session_state.ml_target
    ):

        best_name = get_best_model_name()
        best_model = st.session_state.ml_models.get(
            best_name
        )

        new_mix = pd.DataFrame([{
            "GGBS (kg/m³)": pred_ggbs,
            "NaOH Molarity (M)": pred_naoh,
            "Na₂SiO₃/NaOH Ratio": pred_sio,
            "Activator/Binder Ratio": pred_ab,
            "Water/Binder Ratio": pred_wb,
            "Fine Aggregate (kg/m³)": pred_fine,
            "Coarse Aggregate (kg/m³)": pred_coarse,
            "Superplasticizer (kg/m³)": pred_sp,
            "Curing Temperature (°C)": pred_temp,
            "Curing Duration (h)": pred_duration,
            "Slump (mm)": pred_slump,
            "Density (kg/m³)": pred_density
        }])

        new_mix = new_mix[
            st.session_state.ml_features
        ]

        # Detect extrapolation beyond the laboratory dataset used for training.
        range_warnings = []
        if st.session_state.ml_full_X is not None:
            for col in st.session_state.ml_features:
                if col in new_mix.columns and col in st.session_state.ml_full_X.columns:
                    vals = pd.to_numeric(st.session_state.ml_full_X[col], errors="coerce").dropna()
                    if not vals.empty:
                        v = float(new_mix.iloc[0][col])
                        if v < vals.min() or v > vals.max():
                            range_warnings.append(f"{col}: {v:.3g} is outside training range [{vals.min():.3g}, {vals.max():.3g}]")
        if range_warnings:
            st.warning("⚠️ Extrapolation warning — the prediction is outside the observed training-data range:")
            for msg in range_warnings:
                st.write("• " + msg)

        if st.button(
            f"🔮 Predict {prediction_age}-Day Strength",
            type="primary"
        ):

            value = float(
                best_model.predict(new_mix)[0]
            )

            best_row = st.session_state.ml_results[
                st.session_state.ml_results["Model"] == best_name
            ].iloc[0]
            cv_rmse = float(best_row.get("CV RMSE (MPa)", np.nan))

            st.success(
                f"{best_name} predicted {prediction_age}-day "
                f"strength: {value:.2f} MPa"
            )
            if np.isfinite(cv_rmse):
                st.info(f"Typical cross-validation error: ±{cv_rmse:.2f} MPa. This is an error indicator, not a certified confidence interval.")

            st.metric(
                "Predicted Compressive Strength",
                f"{value:.2f} MPa"
            )

    else:

        st.warning(
            f"Train a model with {age_target} as target first."
        )


# =========================================================
# LAB DATA
# =========================================================

elif page == "🧪 Lab Data":

    st.title("🧪 Experimental & Laboratory Data")

    st.info(
        "Actual laboratory observations belong here. "
        "Demo/synthetic values must remain clearly separated."
    )

    if st.session_state.ml_dataset is not None:
        st.dataframe(
            st.session_state.ml_dataset.head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "No dataset is currently stored in this session."
        )

    st.caption(
        "Future laboratory records can be added and used for model retraining."
    )


# =========================================================
# AI / XGBOOST
# =========================================================

elif page == "🤖 AI / XGBoost":

    st.title("🤖 AI / Machine Learning")
    st.success("Primary model is selected from the evaluated models using test R², with cross-validation reported for robustness.")

    st.header("📂 Dataset Upload")

    uploaded_file = st.file_uploader(
        "Upload Demo or Real Experimental Dataset",
        type=["xlsx", "csv"],
        key="ml_dataset_uploader"
    )

    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.getvalue()
            uploaded_hash = _file_sha256(file_bytes)
            saved_hash = st.session_state.get("ml_dataset_hash", "")
            saved_name = st.session_state.get("ml_dataset_name", "")

            # Preserve a restored model when the uploaded file is the same
            # dataset. A genuinely new file invalidates the old model/results.
            same_dataset = bool(
                saved_hash and saved_hash == uploaded_hash
            ) or (
                not saved_hash
                and saved_name
                and saved_name == uploaded_file.name
                and bool(st.session_state.get("ml_models"))
            )

            if not same_dataset:
                clear_ml_training_state()
                st.session_state.ml_restore_status = (
                    "New dataset detected. Train the models for this dataset before running Mix Optimization."
                )

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(
                    io.BytesIO(file_bytes),
                    sheet_name=0
                )

            st.session_state.ml_dataset = df
            st.session_state.ml_dataset_name = uploaded_file.name
            st.session_state.ml_dataset_hash = uploaded_hash

            st.success(
                f"Dataset loaded successfully: {uploaded_file.name}"
            )
            if same_dataset and st.session_state.ml_models:
                st.success(
                    "Previously trained models were restored automatically for this dataset. "
                    "Mix Optimization is ready without retraining."
                )
            elif st.session_state.get("ml_restore_status"):
                st.info(st.session_state.ml_restore_status)

            st.write(
                f"Records available: **{len(df)}**"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

            available_targets = [
                c for c in TARGET_OPTIONS
                if c in df.columns
            ]

            if not available_targets:

                st.error(
                    "No strength target columns found."
                )

            else:

                target = st.selectbox(
                    "Select Prediction Target",
                    available_targets,
                    index=(
                        available_targets.index(
                            "28-Day Strength (MPa)"
                        )
                        if "28-Day Strength (MPa)" in available_targets
                        else 0
                    ),
                    key="target_selector"
                )

                st.session_state.ml_target = target

                available_features = [
                    c for c in FEATURE_CANDIDATES
                    if c in df.columns
                ]

                st.header("🔢 ML Features")

                st.write(
                    f"Features detected: **{len(available_features)}**"
                )

                st.dataframe(
                    pd.DataFrame({
                        "Feature": available_features
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                if len(available_features) < 3:

                    st.error(
                        "Not enough ML features."
                    )

                else:

                    clean = df[
                        available_features + [target]
                    ].copy()

                    for col in available_features:
                        clean[col] = pd.to_numeric(
                            clean[col],
                            errors="coerce"
                        )

                    clean[target] = pd.to_numeric(
                        clean[target],
                        errors="coerce"
                    )

                    valid_rows = clean.dropna(
                        subset=[target]
                    )

                    st.write(
                        f"Valid target rows: **{len(valid_rows)}**"
                    )

                    if len(valid_rows) < 20:

                        st.warning(
                            "At least 20 valid records are recommended."
                        )

                    else:

                        try:

                            (
                                trained_target,
                                features,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                results,
                                pred_table,
                                trained_models,
                                X,
                                y
                            ) = train_model_bundle(df)

                            # Do not replace state until training is explicitly clicked.
                            if st.button(
                                "🚀 TRAIN & COMPARE MODELS",
                                type="primary"
                            ):

                                set_ml_state(
                                    uploaded_file.name,
                                    trained_target,
                                    features,
                                    X_train,
                                    X_test,
                                    y_train,
                                    y_test,
                                    results,
                                    pred_table,
                                    trained_models,
                                    X,
                                    y,
                                    df
                                )

                                # Recalculate dependent outputs from the current
                                # recommended mix only when explicitly requested later.
                                st.success(
                                    "Model training and comparison completed and saved. "
                                    "The trained models will be restored after an app restart."
                                )

                            if st.session_state.ml_results is not None:

                                if st.session_state.get("ml_restore_status") and st.session_state.get("ml_models"):
                                    st.info(st.session_state.ml_restore_status)

                                st.header("🏆 Model Comparison")

                                display = (
                                    st.session_state.ml_results[
                                        [
                                            "Rank",
                                            "Model",
                                            "R²",
                                            "MAE (MPa)",
                                            "RMSE (MPa)",
                                            "CV R² Mean",
                                            "CV R² Std",
                                            "CV RMSE (MPa)",
                                            "Reliability"
                                        ]
                                    ].copy()
                                )

                                display["R²"] = (
                                    display["R²"].round(4)
                                )
                                display["MAE (MPa)"] = (
                                    display["MAE (MPa)"].round(3)
                                )
                                display["RMSE (MPa)"] = (
                                    display["RMSE (MPa)"].round(3)
                                )
                                for col in ["CV R² Mean", "CV R² Std"]:
                                    display[col] = display[col].round(4)
                                display["CV RMSE (MPa)"] = display["CV RMSE (MPa)"].round(3)

                                st.dataframe(
                                    display,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                best = (
                                    st.session_state.ml_results.iloc[0]
                                )

                                st.success(
                                    f"Best model on this test split: "
                                    f"{best['Model']} "
                                    f"(R² = {best['R²']:.4f})"
                                )

                                st.header(
                                    "📈 Actual vs Predicted"
                                )

                                pred = (
                                    st.session_state.ml_predictions
                                )

                                model_options = [
                                    m for m in [
                                        "XGBoost",
                                        "Random Forest",
                                        "SVM",
                                        "ANN"
                                    ]
                                    if m in pred.columns
                                ]

                                if model_options:

                                    selected = st.selectbox(
                                        "Select model for graph",
                                        model_options,
                                        key="graph_model_selector"
                                    )

                                    graph_df = pred[
                                        [selected, "Actual"]
                                    ].sort_values(
                                        "Actual"
                                    )

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x=graph_df["Actual"],
                                            y=graph_df[selected],
                                            mode="markers",
                                            name=selected
                                        )
                                    )

                                    lo = min(
                                        graph_df["Actual"].min(),
                                        graph_df[selected].min()
                                    )

                                    hi = max(
                                        graph_df["Actual"].max(),
                                        graph_df[selected].max()
                                    )

                                    fig.add_trace(
                                        go.Scatter(
                                            x=[lo, hi],
                                            y=[lo, hi],
                                            mode="lines",
                                            name="Ideal 1:1"
                                        )
                                    )

                                    fig.update_layout(
                                        title=(
                                            f"{selected}: "
                                            "Actual vs Predicted"
                                        ),
                                        xaxis_title=(
                                            "Actual Strength (MPa)"
                                        ),
                                        yaxis_title=(
                                            "Predicted Strength (MPa)"
                                        ),
                                        height=450
                                    )

                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True
                                    )

                                    st.header(
                                        "🔍 Feature Analysis"
                                    )

                                    if (
                                        selected == "XGBoost"
                                        and "XGBoost"
                                        in st.session_state.ml_models
                                    ):

                                        pipe = (
                                            st.session_state.ml_models[
                                                "XGBoost"
                                            ]
                                        )
                                        estimator = (
                                            pipe.named_steps["model"]
                                        )

                                        importance_values = (
                                            estimator.feature_importances_
                                        )

                                        importance_method = (
                                            "XGBoost native feature importance"
                                        )

                                    else:

                                        fitted = (
                                            st.session_state.ml_models[
                                                selected
                                            ]
                                        )

                                        perm = permutation_importance(
                                            fitted,
                                            st.session_state.ml_full_X,
                                            st.session_state.ml_full_y,
                                            n_repeats=10,
                                            random_state=42,
                                            scoring="r2"
                                        )

                                        importance_values = (
                                            perm.importances_mean
                                        )

                                        importance_method = (
                                            "Permutation importance"
                                        )

                                    importance = pd.DataFrame({
                                        "Feature": (
                                            st.session_state.ml_features
                                        ),
                                        "Importance": importance_values
                                    }).sort_values(
                                        "Importance",
                                        ascending=False
                                    ).reset_index(drop=True)

                                    importance["Importance"] = (
                                        importance["Importance"].round(4)
                                    )

                                    st.caption(
                                        f"Method: {importance_method}"
                                    )

                                    st.dataframe(
                                        importance,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                    imp_fig = go.Figure()

                                    imp_fig.add_trace(
                                        go.Bar(
                                            x=importance["Importance"],
                                            y=importance["Feature"],
                                            orientation="h"
                                        )
                                    )

                                    imp_fig.update_layout(
                                        title=(
                                            f"{selected} Feature Importance"
                                        ),
                                        xaxis_title="Importance",
                                        yaxis_title="Feature",
                                        height=520
                                    )

                                    st.plotly_chart(
                                        imp_fig,
                                        use_container_width=True
                                    )

                                st.header("📋 Test Predictions")

                                st.dataframe(
                                    pred.round(3),
                                    use_container_width=True,
                                    hide_index=True
                                )

                        except Exception as training_error:

                            st.error(
                                "Training could not be prepared."
                            )
                            st.code(
                                str(training_error)
                            )

        except Exception as error:

            st.error(
                "The dataset could not be read."
            )
            st.code(
                str(error)
            )

    st.divider()

    framework = pd.DataFrame({
        "Model": [
            "XGBoost",
            "Random Forest",
            "Support Vector Machine",
            "Artificial Neural Network",
            "LSTM"
        ],
        "Role": [
            "Primary",
            "Comparison",
            "Comparison",
            "Comparison",
            "Advanced / Planned"
        ],
        "Status": [
            "Training + comparison",
            "Training + comparison",
            "Training + comparison",
            "Training + comparison",
            "Planned advanced stage"
        ]
    })

    st.header("🤖 AI Model Framework")

    st.dataframe(
        framework,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Demo/synthetic data are for software and pipeline testing only. "
        "They are not laboratory findings."
    )


# =========================================================
# =========================================================
# MIX OPTIMIZATION
# =========================================================

elif page == "🎯 Mix Optimization":

    st.title("🎯 Model-Based Mix Optimization")

    if not st.session_state.ml_models:
        st.warning(
            "First train the model in 🤖 AI / XGBoost."
        )

    elif not st.session_state.ml_features:
        st.warning(
            "ML features are not available. Train the model first."
        )

    else:

        best_name = get_best_model_name()
        best_model = st.session_state.ml_models.get(best_name)

        st.success(
            f"Current optimization model: {best_name}"
        )

        st.session_state.optimization_target = st.number_input(
            "Target 28-Day Compressive Strength (MPa)",
            min_value=20.0,
            max_value=100.0,
            value=float(st.session_state.optimization_target),
            step=0.5,
            key="optimization_target_input"
        )

        # Use observed training-data ranges by default. This avoids
        # unnecessary extrapolation beyond the data used to train the model.
        train_X = st.session_state.ml_full_X

        st.header("🔧 Search Ranges")
        st.caption(
            "Defaults are based on the observed training dataset ranges. "
            "Keep the search inside validated/material-feasible limits."
        )

        def data_range(col, fallback_min, fallback_max):
            if train_X is not None and col in train_X.columns:
                vals = pd.to_numeric(train_X[col], errors="coerce").dropna()
                if not vals.empty:
                    return float(vals.min()), float(vals.max())
            return float(fallback_min), float(fallback_max)

        # Data-driven defaults
        ggbs_lo, ggbs_hi = data_range("GGBS (kg/m³)", 350, 450)
        naoh_lo, naoh_hi = data_range("NaOH Molarity (M)", 8, 16)
        sio_lo, sio_hi = data_range("Na₂SiO₃/NaOH Ratio", 1.5, 2.5)
        ab_lo, ab_hi = data_range("Activator/Binder Ratio", 0.35, 0.55)
        wb_lo, wb_hi = data_range("Water/Binder Ratio", 0.30, 0.40)
        sp_lo, sp_hi = data_range("Superplasticizer (kg/m³)", 0, 2)
        temp_lo, temp_hi = data_range("Curing Temperature (°C)", 25, 80)
        dur_lo, dur_hi = data_range("Curing Duration (h)", 24, 72)

        a1, a2, a3 = st.columns(3)

        with a1:
            min_ggbs = st.number_input(
                "GGBS minimum (kg/m³)",
                min_value=100.0,
                max_value=800.0,
                value=ggbs_lo,
                step=5.0,
                key="opt_min_ggbs"
            )
            max_ggbs = st.number_input(
                "GGBS maximum (kg/m³)",
                min_value=100.0,
                max_value=800.0,
                value=ggbs_hi,
                step=5.0,
                key="opt_max_ggbs"
            )

            min_naoh = st.number_input(
                "NaOH minimum (M)",
                min_value=1.0,
                max_value=20.0,
                value=naoh_lo,
                step=0.5,
                key="opt_min_naoh"
            )
            max_naoh = st.number_input(
                "NaOH maximum (M)",
                min_value=1.0,
                max_value=20.0,
                value=naoh_hi,
                step=0.5,
                key="opt_max_naoh"
            )

        with a2:
            min_sio = st.number_input(
                "Silicate/NaOH minimum",
                min_value=0.5,
                max_value=4.0,
                value=sio_lo,
                step=0.05,
                key="opt_min_sio"
            )
            max_sio = st.number_input(
                "Silicate/NaOH maximum",
                min_value=0.5,
                max_value=4.0,
                value=sio_hi,
                step=0.05,
                key="opt_max_sio"
            )

            min_ab = st.number_input(
                "Activator/Binder minimum",
                min_value=0.10,
                max_value=1.00,
                value=ab_lo,
                step=0.01,
                key="opt_min_ab"
            )
            max_ab = st.number_input(
                "Activator/Binder maximum",
                min_value=0.10,
                max_value=1.00,
                value=ab_hi,
                step=0.01,
                key="opt_max_ab"
            )

        with a3:
            min_wb = st.number_input(
                "Water/Binder minimum",
                min_value=0.10,
                max_value=0.80,
                value=wb_lo,
                step=0.01,
                key="opt_min_wb"
            )
            max_wb = st.number_input(
                "Water/Binder maximum",
                min_value=0.10,
                max_value=0.80,
                value=wb_hi,
                step=0.01,
                key="opt_max_wb"
            )

            min_sp = st.number_input(
                "Superplasticizer minimum (kg/m³)",
                min_value=0.0,
                max_value=10.0,
                value=sp_lo,
                step=0.1,
                key="opt_min_sp"
            )
            max_sp = st.number_input(
                "Superplasticizer maximum (kg/m³)",
                min_value=0.0,
                max_value=10.0,
                value=sp_hi,
                step=0.1,
                key="opt_max_sp"
            )

        t1, t2 = st.columns(2)

        with t1:
            min_temp = st.number_input(
                "Curing Temperature minimum (°C)",
                min_value=10.0,
                max_value=100.0,
                value=temp_lo,
                step=1.0,
                key="opt_min_temp"
            )
            max_temp = st.number_input(
                "Curing Temperature maximum (°C)",
                min_value=10.0,
                max_value=100.0,
                value=temp_hi,
                step=1.0,
                key="opt_max_temp"
            )

        with t2:
            min_duration = st.number_input(
                "Curing Duration minimum (h)",
                min_value=1.0,
                max_value=168.0,
                value=dur_lo,
                step=1.0,
                key="opt_min_duration"
            )
            max_duration = st.number_input(
                "Curing Duration maximum (h)",
                min_value=1.0,
                max_value=168.0,
                value=dur_hi,
                step=1.0,
                key="opt_max_duration"
            )

        st.header("📌 Fixed Conditions")

        f1, f2 = st.columns(2)

        with f1:
            fixed_fine = st.number_input(
                "Fine Aggregate (kg/m³)",
                min_value=400.0,
                max_value=1200.0,
                value=700.0,
                step=10.0,
                key="opt_fixed_fine"
            )
            fixed_coarse = st.number_input(
                "Coarse Aggregate (kg/m³)",
                min_value=600.0,
                max_value=1600.0,
                value=1100.0,
                step=10.0,
                key="opt_fixed_coarse"
            )

        with f2:
            fixed_slump = st.number_input(
                "Slump (mm)",
                min_value=0.0,
                max_value=300.0,
                value=100.0,
                step=1.0,
                key="opt_fixed_slump"
            )
            fixed_density = st.number_input(
                "Density (kg/m³)",
                min_value=1800.0,
                max_value=2600.0,
                value=2350.0,
                step=10.0,
                key="opt_fixed_density"
            )

        st.caption(
            "Fine aggregate, coarse aggregate, slump and density are held fixed "
            "in this optimization prototype; the main process/mix controls are varied."
        )
        st.info(
            "Optimization now uses a Multi-Criteria Performance Score (MCPS): "
            "40% strength, 20% workability, 20% relative cost and 20% relative CO₂. "
            "A candidate must first satisfy the strength target to be considered a target-achieving recommendation."
        )

        candidate_count = st.slider(
            "Number of candidate mixes",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500,
            key="opt_candidate_count"
        )

        if st.button(
            "🎯 SEARCH FOR BEST MIX",
            type="primary",
            key="search_best_mix"
        ):

            invalid = (
                min_ggbs > max_ggbs
                or min_naoh > max_naoh
                or min_sio > max_sio
                or min_ab > max_ab
                or min_wb > max_wb
                or min_sp > max_sp
                or min_temp > max_temp
                or min_duration > max_duration
            )

            if invalid:
                st.error("Check that every minimum is less than or equal to its maximum.")

            else:

                rng = np.random.default_rng(42)

                candidates = pd.DataFrame({
                    "GGBS (kg/m³)": rng.uniform(min_ggbs, max_ggbs, candidate_count),
                    "NaOH Molarity (M)": rng.uniform(min_naoh, max_naoh, candidate_count),
                    "Na₂SiO₃/NaOH Ratio": rng.uniform(min_sio, max_sio, candidate_count),
                    "Activator/Binder Ratio": rng.uniform(min_ab, max_ab, candidate_count),
                    "Water/Binder Ratio": rng.uniform(min_wb, max_wb, candidate_count),
                    "Fine Aggregate (kg/m³)": fixed_fine,
                    "Coarse Aggregate (kg/m³)": fixed_coarse,
                    "Superplasticizer (kg/m³)": rng.uniform(min_sp, max_sp, candidate_count),
                    "Curing Temperature (°C)": rng.uniform(min_temp, max_temp, candidate_count),
                    "Curing Duration (h)": rng.uniform(min_duration, max_duration, candidate_count),
                    "Slump (mm)": fixed_slump,
                    "Density (kg/m³)": fixed_density
                })

                prediction_input = candidates[
                    st.session_state.ml_features
                ]

                prediction = best_model.predict(prediction_input)

                candidates["Predicted 28-Day Strength (MPa)"] = prediction

                target_value = float(st.session_state.optimization_target)

                candidates["Strength Gap to Target (MPa)"] = (
                    candidates["Predicted 28-Day Strength (MPa)"] - target_value
                )

                candidates["Target Achieved"] = (
                    candidates["Predicted 28-Day Strength (MPa)"] >= target_value
                )

                candidates["Absolute Error vs Target (MPa)"] = abs(
                    candidates["Strength Gap to Target (MPa)"]
                )

                # Multi-criteria decision score: prioritize target achievement,
                # then workability, material cost and indicative CO₂.
                total_activator = candidates["GGBS (kg/m³)"] * candidates["Activator/Binder Ratio"]
                candidates["NaOH (kg/m³)"] = total_activator / (1.0 + candidates["Na₂SiO₃/NaOH Ratio"])
                candidates["Sodium Silicate (kg/m³)"] = total_activator - candidates["NaOH (kg/m³)"]
                candidates["Estimated Cost (₹/m³)"] = (
                    candidates["GGBS (kg/m³)"] * st.session_state.cost_rates["GGBS"]
                    + candidates["NaOH (kg/m³)"] * st.session_state.cost_rates["NaOH"]
                    + candidates["Sodium Silicate (kg/m³)"] * st.session_state.cost_rates["Sodium Silicate"]
                    + candidates["Fine Aggregate (kg/m³)"] * st.session_state.cost_rates["Fine Aggregate"]
                    + candidates["Coarse Aggregate (kg/m³)"] * st.session_state.cost_rates["Coarse Aggregate"]
                    + candidates["Superplasticizer (kg/m³)"] * st.session_state.cost_rates["Superplasticizer"]
                )
                candidates["Indicative CO₂ (kg/m³)"] = (
                    candidates["GGBS (kg/m³)"] * st.session_state.co2_factors["GGBS"]
                    + candidates["NaOH (kg/m³)"] * st.session_state.co2_factors["NaOH"]
                    + candidates["Sodium Silicate (kg/m³)"] * st.session_state.co2_factors["Sodium Silicate"]
                    + candidates["Fine Aggregate (kg/m³)"] * st.session_state.co2_factors["Fine Aggregate"]
                    + candidates["Coarse Aggregate (kg/m³)"] * st.session_state.co2_factors["Coarse Aggregate"]
                    + candidates["Superplasticizer (kg/m³)"] * st.session_state.co2_factors["Superplasticizer"]
                )
                candidates["Workability Score"] = np.where(
                    candidates["Slump (mm)"].between(75, 150),
                    100.0,
                    np.maximum(0.0, 100.0 - np.minimum(
                        abs(candidates["Slump (mm)"] - 75),
                        abs(candidates["Slump (mm)"] - 150)
                    ) * 2.0)
                )
                strength_component = np.clip(
                    candidates["Predicted 28-Day Strength (MPa)"] / max(target_value, 1e-9) * 100.0, 0, 100
                )
                cost_component = (candidates["Estimated Cost (₹/m³)"].min() / candidates["Estimated Cost (₹/m³)"]) * 100.0
                co2_component = (candidates["Indicative CO₂ (kg/m³)"].min() / candidates["Indicative CO₂ (kg/m³)"]) * 100.0
                candidates["MCPS"] = (
                    0.40 * strength_component
                    + 0.20 * candidates["Workability Score"]
                    + 0.20 * cost_component
                    + 0.20 * co2_component
                )

                # 1) Closest candidate: minimum absolute difference from target.
                closest_ranked = candidates.sort_values(
                    by=[
                        "Absolute Error vs Target (MPa)",
                        "Predicted 28-Day Strength (MPa)"
                    ],
                    ascending=[True, True]
                ).reset_index(drop=True)

                closest_candidate = closest_ranked.iloc[0].copy()

                # 2) Highest-strength candidate: maximum predicted strength.
                higher_ranked = candidates.sort_values(
                    by="Predicted 28-Day Strength (MPa)",
                    ascending=False
                ).reset_index(drop=True)

                higher_candidate = higher_ranked.iloc[0].copy()

                # Save both candidates so all downstream modules can use them.
                st.session_state.optimization_results = closest_ranked
                st.session_state.optimization_closest_mix = closest_candidate
                st.session_state.optimization_higher_mix = higher_candidate
                st.session_state.optimization_target_met_closest = bool(
                    closest_candidate["Target Achieved"]
                )
                st.session_state.optimization_target_met_higher = bool(
                    higher_candidate["Target Achieved"]
                )

                # Primary recommendation uses multi-criteria ranking among feasible
                # target-achieving candidates. If none reaches target, choose the
                # closest candidate and explicitly mark target failure.
                feasible = candidates[candidates["Target Achieved"]].copy()
                if not feasible.empty:
                    primary_candidate = feasible.sort_values(
                        by=["MCPS", "Absolute Error vs Target (MPa)"],
                        ascending=[False, True]
                    ).iloc[0].copy()
                    primary_label = "Best Multi-Criteria Target-Achieving Mix"
                else:
                    primary_candidate = closest_candidate
                    primary_label = "Closest Mix (Target Not Achieved)"

                st.session_state.recommended_mix = primary_candidate
                st.session_state.recommended_mix_label = primary_label

                # Use the closest-to-target ranking for the Top 10 table/chart.
                # This keeps the comparison centered on the selected target.
                ranked = closest_ranked.copy()

        if st.session_state.recommended_mix is not None:

            rec = st.session_state.recommended_mix
            closest = st.session_state.optimization_closest_mix
            higher = st.session_state.optimization_higher_mix

            # Rebuild the ranking from saved optimization results so this
            # section never depends on a local variable from an earlier branch.
            if st.session_state.optimization_results is not None:
                ranked = st.session_state.optimization_results.copy()
            else:
                ranked = pd.DataFrame([closest, higher])

            st.header("🏆 Optimization Results")

            # Show both requested candidates side-by-side.
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("🎯 Closest to Target")
                st.metric(
                    "Predicted Strength",
                    f"{float(closest['Predicted 28-Day Strength (MPa)']):.2f} MPa"
                )
                st.metric(
                    "Difference from Target",
                    f"{abs(float(closest['Predicted 28-Day Strength (MPa)']) - float(st.session_state.optimization_target)):.2f} MPa"
                )
                st.metric(
                    "Target Achieved",
                    "YES" if bool(closest["Target Achieved"]) else "NO"
                )

            with c2:
                st.subheader("🚀 Highest Predicted Strength")
                st.metric(
                    "Predicted Strength",
                    f"{float(higher['Predicted 28-Day Strength (MPa)']):.2f} MPa"
                )
                st.metric(
                    "Difference from Target",
                    f"{abs(float(higher['Predicted 28-Day Strength (MPa)']) - float(st.session_state.optimization_target)):.2f} MPa"
                )
                st.metric(
                    "Target Achieved",
                    "YES" if bool(higher["Target Achieved"]) else "NO"
                )

            st.info(
                f"Target = {float(st.session_state.optimization_target):.2f} MPa. "
                f"Primary recommendation: {st.session_state.recommended_mix_label}."
            )

            recommendation = pd.DataFrame({
                "Parameter": [
                    "Selection",
                    "Best ML Model",
                    "GGBS",
                    "NaOH Molarity",
                    "Silicate/NaOH Ratio",
                    "Activator/Binder Ratio",
                    "Water/Binder Ratio",
                    "Fine Aggregate",
                    "Coarse Aggregate",
                    "Superplasticizer",
                    "Curing Temperature",
                    "Curing Duration",
                    "Slump",
                    "Density",
                    "Predicted 28-Day Strength",
                    "Target",
                    "Target Achieved"
                ],
                "Value": [
                    st.session_state.recommended_mix_label,
                    best_name,
                    f"{float(rec['GGBS (kg/m³)']):.2f} kg/m³",
                    f"{float(rec['NaOH Molarity (M)']):.2f} M",
                    f"{float(rec['Na₂SiO₃/NaOH Ratio']):.3f}",
                    f"{float(rec['Activator/Binder Ratio']):.3f}",
                    f"{float(rec['Water/Binder Ratio']):.3f}",
                    f"{float(rec['Fine Aggregate (kg/m³)']):.2f} kg/m³",
                    f"{float(rec['Coarse Aggregate (kg/m³)']):.2f} kg/m³",
                    f"{float(rec['Superplasticizer (kg/m³)']):.2f} kg/m³",
                    f"{float(rec['Curing Temperature (°C)']):.2f} °C",
                    f"{float(rec['Curing Duration (h)']):.1f} h",
                    f"{float(rec['Slump (mm)']):.1f} mm",
                    f"{float(rec['Density (kg/m³)']):.1f} kg/m³",
                    f"{float(rec['Predicted 28-Day Strength (MPa)']):.2f} MPa",
                    f"{float(st.session_state.optimization_target):.2f} MPa",
                    "YES" if bool(rec["Target Achieved"]) else "NO"
                ]
            })

            st.header("🏆 Primary Recommended Mix")
            st.dataframe(
                recommendation,
                use_container_width=True,
                hide_index=True
            )

            if not bool(closest["Target Achieved"]) and not bool(higher["Target Achieved"]):
                st.warning(
                    "Neither the closest candidate nor the highest-strength candidate "
                    "reaches the target within the current search. Keep both results for "
                    "comparison, but do not treat either as a final validated mix."
                )
            elif bool(closest["Target Achieved"]):
                st.success(
                    "The closest candidate already achieves the target. "
                    "It is the primary recommendation for the current search."
                )
            else:
                st.success(
                    "The highest-strength candidate achieves the target. "
                    "It is the primary recommendation because the closest candidate did not."
                )

            st.header("📋 Top 10 Candidate Mixes")

            st.dataframe(
                ranked.head(10).round(3),
                use_container_width=True,
                hide_index=True
            )

            st.header("📈 Candidate Strengths")

            plot_df = ranked.head(20).copy()

            opt_fig = go.Figure()

            opt_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(plot_df) + 1)),
                    y=plot_df["Predicted 28-Day Strength (MPa)"],
                    mode="lines+markers",
                    name="Predicted Strength"
                )
            )

            opt_fig.add_hline(
                y=float(st.session_state.optimization_target),
                line_dash="dash",
                annotation_text="Target"
            )

            opt_fig.update_layout(
                title="Top Candidate Mix Predictions",
                xaxis_title="Candidate Rank",
                yaxis_title="Predicted 28-Day Strength (MPa)",
                height=420
            )

            st.plotly_chart(
                opt_fig,
                use_container_width=True
            )

            st.warning(
                "Optimization output is model-based. The recommended candidate "
                "must be checked against material availability, mass/volume balance, "
                "workability and laboratory testing before being treated as a final mix."
            )

# COST ANALYSIS
# =========================================================

elif page == "💰 Cost Analysis":

    st.title("💰 Cost Analysis")

    st.info(
        "Indicative material cost per m³. Enter current supplier/local "
        "rates for a project-specific estimate."
    )

    st.header("💵 Material Prices")

    c1, c2 = st.columns(2)

    with c1:
        st.session_state.cost_rates["GGBS"] = st.number_input(
            "GGBS price (₹/kg)",
            0.0,
            1000.0,
            float(st.session_state.cost_rates["GGBS"]),
            0.5
        )

        st.session_state.cost_rates["NaOH"] = st.number_input(
            "NaOH price (₹/kg)",
            0.0,
            1000.0,
            float(st.session_state.cost_rates["NaOH"]),
            0.5
        )

        st.session_state.cost_rates["Sodium Silicate"] = st.number_input(
            "Sodium Silicate price (₹/kg)",
            0.0,
            1000.0,
            float(st.session_state.cost_rates["Sodium Silicate"]),
            0.5
        )

    with c2:
        st.session_state.cost_rates["Fine Aggregate"] = st.number_input(
            "Fine Aggregate price (₹/kg)",
            0.0,
            1000.0,
            float(st.session_state.cost_rates["Fine Aggregate"]),
            0.05
        )

        st.session_state.cost_rates["Coarse Aggregate"] = st.number_input(
            "Coarse Aggregate price (₹/kg)",
            0.0,
            1000.0,
            float(st.session_state.cost_rates["Coarse Aggregate"]),
            0.05
        )

        st.session_state.cost_rates["Superplasticizer"] = st.number_input(
            "Superplasticizer price (₹/kg)",
            0.0,
            2000.0,
            float(st.session_state.cost_rates["Superplasticizer"]),
            1.0
        )

    if st.session_state.recommended_mix is None:

        st.warning(
            "Run 🎯 Mix Optimization first."
        )

    else:

        mix = st.session_state.recommended_mix
        mix_label = st.session_state.get(
            "recommended_mix_label",
            "Primary Recommended Mix"
        )

        st.success(
            f"{mix_label} is selected automatically from Mix Optimization."
        )

        st.caption(
            f"Optimization prediction: {float(mix['Predicted 28-Day Strength (MPa)']):.2f} MPa "
            f"| Target: {float(st.session_state.optimization_target):.2f} MPa"
        )

        cost_df, total = calculate_cost_for_mix(mix)

        st.session_state.cost_result = {
            "total": total,
            "breakdown": cost_df,
            "mix_label": mix_label,
            "predicted_strength": float(
                mix["Predicted 28-Day Strength (MPa)"]
            )
        }

        save_project_state()

        st.header("📋 Cost Breakdown")

        st.dataframe(
            cost_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Estimated Material Cost",
            f"₹{total:,.2f} / m³"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=cost_df["Material"],
                y=cost_df["Cost (₹/m³)"]
            )
        )

        fig.update_layout(
            title="Material-wise Cost Contribution",
            xaxis_title="Material",
            yaxis_title="Cost (₹/m³)",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# CO2 ANALYSIS
# =========================================================

elif page == "🌱 CO₂ Analysis":

    st.title("🌱 CO₂ Analysis")

    st.info(
        "Indicative material CO₂ estimate. Emission factors are "
        "editable assumptions and must be replaced with verified "
        "supplier/LCA data for final reporting."
    )

    st.header("⚙️ Emission Factors")

    a, b = st.columns(2)

    with a:
        for material in [
            "GGBS",
            "NaOH",
            "Sodium Silicate"
        ]:
            st.session_state.co2_factors[material] = st.number_input(
                f"{material} emission factor (kg CO₂e/kg)",
                0.0,
                10.0,
                float(
                    st.session_state.co2_factors[material]
                ),
                0.01,
                key=f"co2_{material}"
            )

    with b:
        for material in [
            "Fine Aggregate",
            "Coarse Aggregate",
            "Superplasticizer"
        ]:
            st.session_state.co2_factors[material] = st.number_input(
                f"{material} emission factor (kg CO₂e/kg)",
                0.0,
                10.0,
                float(
                    st.session_state.co2_factors[material]
                ),
                0.001,
                key=f"co2_{material}"
            )

    if st.session_state.recommended_mix is None:

        st.warning(
            "Run 🎯 Mix Optimization first."
        )

    else:

        st.success(
            f"{st.session_state.get('recommended_mix_label', 'Primary Recommended Mix')} "
            "is automatically selected from Mix Optimization."
        )

        co2_df, total = calculate_co2_for_mix(
            st.session_state.recommended_mix
        )

        st.session_state.co2_result = {
            "total": total,
            "breakdown": co2_df
        }

        save_project_state()

        st.header("📋 CO₂ Breakdown")

        st.dataframe(
            co2_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Indicative Material CO₂",
            f"{total:.2f} kg CO₂e / m³"
        )

        st.caption(
            "Illustrative factors are for software testing until "
            "verified LCA/supplier values are entered."
        )


# =========================================================
# QUALITY SCORE
# =========================================================

elif page == "⭐ Performance Score":

    st.title("⭐ Multi-Criteria Performance Score (MCPS)")

    st.info(
        "Transparent decision-support score. This is not an "
        "established concrete standard."
    )

    if st.session_state.recommended_mix is None:

        st.warning(
            "Run Mix Optimization first."
        )

    elif (
        st.session_state.cost_result is None
        or st.session_state.co2_result is None
    ):

        st.warning(
            "Run Cost Analysis and CO₂ Analysis first."
        )

    else:

        st.header("⚖️ Score Weights")

        weights = {
            "Strength": 40.0,
            "Workability": 20.0,
            "Cost": 20.0,
            "CO₂": 20.0
        }

        st.write(
            "Strength 40% + Workability 20% + Cost 20% + CO₂ 20% = 100%"
        )

        quality = calculate_quality()

        st.session_state.quality_result = quality

        save_project_state()

        st.header("📊 Component Scores")

        components = pd.DataFrame({
            "Component": [
                "Strength",
                "Workability",
                "Cost",
                "CO₂"
            ],
            "Score (/100)": [
                quality["strength_score"],
                quality["workability_score"],
                quality["cost_score"],
                quality["co2_score"]
            ],
            "Weight (%)": [
                weights["Strength"],
                weights["Workability"],
                weights["Cost"],
                weights["CO₂"]
            ]
        })

        st.dataframe(
            components.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Overall MCPS",
            f"{quality['overall']:.1f} / 100"
        )

        st.caption(
            "Reference values and weights are transparent project "
            "assumptions and should be refined for the final research version."
        )


# =========================================================
# SMART RECOMMENDATION
# =========================================================

elif page == "🧠 Smart Recommendation":

    st.title("🧠 Smart Recommendation")

    st.session_state.smart_result = (
        build_smart_recommendation()
    )

    save_project_state()

    result = st.session_state.smart_result

    st.header("🎯 Smart Decision")

    if result["status"] == "PRELIMINARILY RECOMMENDED":
        st.success(
            f"{result['status']}\n\n{result['message']}"
        )
    else:
        st.warning(
            f"{result['status']}\n\n{result['message']}"
        )

    if result["attention"]:

        st.header("⚠️ Factors Requiring Attention")

        for item in result["attention"]:
            st.write(f"• {item}")

    if result["actions"]:

        st.header("🧪 Required Validation / Next Actions")

        for item in result["actions"]:
            st.write(f"• {item}")

    if st.session_state.recommended_mix is not None:

        st.header("🏆 Recommended Mix")

        rec = st.session_state.recommended_mix

        rec_df = pd.DataFrame({
            "Parameter": [
                "Best ML Model",
                "GGBS",
                "NaOH Molarity",
                "Silicate/NaOH Ratio",
                "Activator/Binder Ratio",
                "Water/Binder Ratio",
                "Fine Aggregate",
                "Coarse Aggregate",
                "Superplasticizer",
                "Curing Temperature",
                "Curing Duration",
                "Predicted 28-Day Strength"
            ],
            "Value": [
                get_best_model_name() or "N/A",
                f"{rec['GGBS (kg/m³)']:.2f} kg/m³",
                f"{rec['NaOH Molarity (M)']:.2f} M",
                f"{rec['Na₂SiO₃/NaOH Ratio']:.3f}",
                f"{rec['Activator/Binder Ratio']:.3f}",
                f"{rec['Water/Binder Ratio']:.3f}",
                f"{rec['Fine Aggregate (kg/m³)']:.2f} kg/m³",
                f"{rec['Coarse Aggregate (kg/m³)']:.2f} kg/m³",
                f"{rec['Superplasticizer (kg/m³)']:.2f} kg/m³",
                f"{rec['Curing Temperature (°C)']:.2f} °C",
                f"{rec['Curing Duration (h)']:.1f} h",
                f"{rec['Predicted 28-Day Strength (MPa)']:.2f} MPa"
            ]
        })

        st.dataframe(
            rec_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Smart Recommendation is a decision-support layer, not a substitute "
        "for laboratory testing, structural assessment or engineering approval."
    )


# =========================================================
# CRACK DETECTION
# =========================================================

elif page == "📷 Crack Detection":

    st.title("📷 Crack Detection")

    st.info(
        "Preliminary image screening only. This prototype uses "
        "image-processing heuristics and is not a trained structural "
        "crack classifier or engineering safety assessment."
    )

    uploaded_image = st.file_uploader(
        "Upload Concrete Crack Image",
        type=["jpg", "jpeg", "png"],
        key="crack_image_uploader"
    )

    if uploaded_image is not None:

        if not PIL_AVAILABLE:

            st.error(
                "Pillow is not available."
            )

        else:

            image = Image.open(uploaded_image)

            st.image(
                image,
                caption="Uploaded concrete image",
                use_container_width=True
            )

            if st.button(
                "🔍 ANALYSE CRACK IMAGE",
                type="primary"
            ):

                (
                    score,
                    area,
                    screening,
                    severity,
                    mask
                ) = analyze_crack_image(image)

                st.session_state.crack_score = score
                st.session_state.crack_area = area
                st.session_state.crack_screening = screening
                st.session_state.crack_severity = severity
                st.session_state.crack_image_name = uploaded_image.name

                save_project_state()

                st.success(
                    "Preliminary crack screening completed."
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Detection Score",
                    f"{score:.1f}/100"
                )

                m2.metric(
                    "Estimated Crack Area",
                    f"{area:.2f}%"
                )

                m3.metric(
                    "Screening",
                    screening
                )

                m4.metric(
                    "Severity",
                    severity
                )

                mask_image = Image.fromarray(
                    (mask.astype(np.uint8) * 255)
                )

                st.subheader(
                    "Crack-Like Region Map"
                )

                st.image(
                    mask_image,
                    caption="Thresholded crack-like region map",
                    use_container_width=True
                )

    elif st.session_state.crack_score is not None:

        st.subheader(
            "Latest Saved Crack Screening"
        )

        saved_crack = pd.DataFrame({
            "Parameter": [
                "Detection Score",
                "Estimated Crack Area",
                "Screening",
                "Severity"
            ],
            "Value": [
                f"{st.session_state.crack_score:.1f}/100",
                f"{st.session_state.crack_area:.2f}%",
                st.session_state.crack_screening,
                st.session_state.crack_severity
            ]
        })

        st.dataframe(
            saved_crack,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# RESULTS & ANALYSIS
# =========================================================

elif page == "📊 Results & Analysis":

    st.title("📊 Results & Analysis")

    if st.session_state.ml_results is not None:

        st.header("🏆 ML Model Performance")

        st.dataframe(
            st.session_state.ml_results.round(4),
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.recommended_mix is not None:

        st.header("🎯 Optimization Result")

        st.write(
            f"Recommended prediction: "
            f"**{get_recommended_mix_strength():.2f} MPa**"
        )

        st.write(
            f"Target: "
            f"**{st.session_state.optimization_target:.2f} MPa**"
        )

    if st.session_state.cost_result is not None:

        st.metric(
            "Material Cost",
            f"₹{st.session_state.cost_result['total']:,.2f}/m³"
        )

    if st.session_state.co2_result is not None:

        st.metric(
            "Indicative CO₂",
            f"{st.session_state.co2_result['total']:.2f} kg/m³"
        )

    if st.session_state.quality_result is not None:

        st.metric(
            "MCPS",
            f"{st.session_state.quality_result['overall']:.1f}/100"
        )

    if st.session_state.crack_score is not None:

        st.header("📷 Crack Screening")

        st.write(
            f"Score: **{st.session_state.crack_score:.1f}/100**"
        )

        st.write(
            f"Area: **{st.session_state.crack_area:.2f}%**"
        )

        st.write(
            f"Screening: **{st.session_state.crack_screening}**"
        )

        st.write(
            f"Severity: **{st.session_state.crack_severity}**"
        )

    if st.session_state.smart_result is not None:

        st.header("🧠 Smart Recommendation")

        st.write(
            st.session_state.smart_result["status"]
        )


# =========================================================
# PDF REPORT
# =========================================================

elif page == "📄 PDF Report":

    st.title("📄 GeoPolymer AI Report")

    st.info(
        "Both report formats are generated from one MASTER PROJECT STATE. "
        "This prevents stale crack/CO₂/quality/recommendation values."
    )

    # Refresh downstream calculations from the same current project state.
    # This prevents reports from showing "not calculated" when the recommended
    # mix already exists but the user has not manually opened every analysis page.
    if st.session_state.get("recommended_mix") is not None:
        try:
            if st.session_state.get("cost_result") is None:
                mix = st.session_state.recommended_mix
                cost_df, total_cost = calculate_cost_for_mix(mix)
                st.session_state.cost_result = {
                    "total": total_cost,
                    "breakdown": cost_df,
                    "mix_label": st.session_state.get(
                        "recommended_mix_label", "Primary Recommended Mix"
                    ),
                    "predicted_strength": float(
                        mix["Predicted 28-Day Strength (MPa)"]
                    )
                }

            if st.session_state.get("co2_result") is None:
                co2_df, total_co2 = calculate_co2_for_mix(
                    st.session_state.recommended_mix
                )
                st.session_state.co2_result = {
                    "total": total_co2,
                    "breakdown": co2_df
                }

            if st.session_state.get("quality_result") is None:
                st.session_state.quality_result = calculate_quality()

            st.session_state.smart_result = build_smart_recommendation()
        except Exception as report_calc_error:
            st.warning(
                "Some downstream report calculations could not be refreshed: "
                + str(report_calc_error)
            )

    save_project_state()

    saved_time = "Not saved"
    if STATE_FILE.exists():
        try:
            state_preview = json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )
            saved_time = state_preview.get("saved_at", "Unknown")
        except Exception:
            saved_time = "Unknown"

    st.success(
        f"MASTER PROJECT STATE ACTIVE | Last saved: {saved_time}"
    )

    # Build both versions from exactly the same source state.
    standard_sections = build_standard_sections()
    descriptive_sections = build_descriptive_sections()

    standard_text = "\n\n".join(
        [heading + "\n" + "\n".join(body) for heading, body in standard_sections]
    )

    descriptive_text = "\n\n".join(
        [heading + "\n" + "\n".join(body) for heading, body in descriptive_sections]
    )

    st.subheader("✅ Current Report Data Check")

    rec = st.session_state.get("recommended_mix")
    cost = st.session_state.get("cost_result")
    co2 = st.session_state.get("co2_result")
    quality = st.session_state.get("quality_result")

    report_check = pd.DataFrame({
        "Item": [
            "Best ML Model",
            "Optimization Target",
            "Primary Predicted Strength",
            "Material Cost",
            "CO₂",
            "MCPS",
            "Crack Detection Score",
            "Estimated Crack Area",
            "Crack Screening",
            "Crack Severity",
            "Smart Recommendation"
        ],
        "Value": [
            get_best_model_name(),
            f"{float(st.session_state.get('optimization_target', 40.0)):.2f} MPa",
            (
                f"{float(rec['Predicted 28-Day Strength (MPa)']):.2f} MPa"
                if rec is not None else "Not available"
            ),
            (
                f"₹{float(cost['total']):,.2f}/m³"
                if cost else "Not calculated"
            ),
            (
                f"{float(co2['total']):.2f} kg CO₂e/m³"
                if co2 else "Not calculated"
            ),
            (
                f"{float(quality['overall']):.1f}/100"
                if quality else "Not calculated"
            ),
            (
                f"{float(st.session_state.crack_score):.1f}/100"
                if st.session_state.crack_score is not None else "Not available"
            ),
            (
                f"{float(st.session_state.crack_area):.2f}%"
                if st.session_state.crack_area is not None else "Not available"
            ),
            st.session_state.get("crack_screening") or "Not available",
            st.session_state.get("crack_severity") or "Not available",
            st.session_state.smart_result.get("status", "Not available")
            if st.session_state.get("smart_result") else "Not available"
        ]
    })

    st.dataframe(
        report_check,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📄 Standard Project Summary Preview")
    st.text_area(
        "Standard report",
        standard_text,
        height=420,
        key="standard_report_preview_final"
    )

    st.subheader("📘 Descriptive Academic / Engineering Preview")
    st.text_area(
        "Descriptive report",
        descriptive_text,
        height=520,
        key="descriptive_report_preview_final"
    )

    try:
        standard_pdf = make_simple_pdf(
            "GEOPOLYMER AI\nPROJECT SUMMARY REPORT",
            standard_sections,
            "Standard Project Summary Report"
        )

        descriptive_pdf = make_simple_pdf(
            "GEOPOLYMER AI\nDESCRIPTIVE ACADEMIC / ENGINEERING REPORT",
            descriptive_sections,
            "Descriptive Academic Report"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.success("Standard PDF ready.")
            st.download_button(
                "📄 Download Standard Project Summary PDF",
                data=standard_pdf,
                file_name="GeoPolymer_AI_Standard_Project_Report.pdf",
                mime="application/pdf",
                key="download_standard_pdf_final"
            )

        with c2:
            st.success("Descriptive Academic PDF ready.")
            st.download_button(
                "📘 Download Descriptive Academic PDF",
                data=descriptive_pdf,
                file_name="GeoPolymer_AI_Descriptive_Academic_Report.pdf",
                mime="application/pdf",
                key="download_descriptive_pdf_final"
            )

        st.caption(
            "PDF generation uses matplotlib/PdfPages and does not require ReportLab."
        )

    except Exception as error:
        st.error("PDF generation could not be completed.")
        st.code(str(error))
        st.info(
            "The text previews above remain available even if PDF creation fails."
        )


# =========================================================
# FINAL AUTO-SAVE
# =========================================================

save_project_state()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "GeoPolymer AI | Geopolymer Concrete Mix Design & "
    "Strength Prediction Platform"
)
