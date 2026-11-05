from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin


def identify_model_type(name, estimator):

    name_text = str(name).lower()
    class_text = estimator.__class__.__name__.lower()

    # Pipeline 내부 모델 확인
    if hasattr(estimator, "named_steps"):
        class_text += " " + " ".join(
            step.__class__.__name__.lower()
            for step in estimator.named_steps.values()
        )

    combined = name_text + " " + class_text

    if "lightgbm" in combined or "lgbm" in combined:
        return "lgbm"

    if "randomforest" in combined or name_text.strip() == "rf":
        return "rf"

    return None


def extract_base_models(model):

    # sklearn VotingClassifier인 경우
    if hasattr(model, "estimators_") and hasattr(model, "estimators"):

        fitted = model.estimators_
        configured = model.estimators

        if len(fitted) == len(configured):

            result = []

            for spec, est in zip(configured, fitted):

                if est is None or est == "drop":
                    continue

                name = spec[0] if isinstance(spec, tuple) else est.__class__.__name__

                result.append((name, est))

            return result

    # Custom VotingEnsembleClassifier인 경우
    estimators = getattr(model, "estimators", None)

    if estimators is None:
        estimators = model.__dict__.get("estimators_", None)

    if estimators is None:
        raise ValueError("Cannot find ensemble estimators.")

    result = []

    for item in estimators:

        if isinstance(item, tuple):
            name, est = item
        else:
            est = item
            name = est.__class__.__name__

        result.append((name, est))

    return result


def weighted_predict_proba(model, X):

    base_models = extract_base_models(model)

    lgbm_prob = None
    rf_prob = None

    for name, estimator in base_models:

        model_type = identify_model_type(name, estimator)

        if model_type is None:
            continue

        # Individual model prediction
        p = estimator.predict_proba(X)

        classes = np.asarray(estimator.classes_)
        positive_idx = np.where(classes == 1)[0]

        if len(positive_idx) != 1:
            raise ValueError(
                f"Positive class 1 not found in {name}."
            )

        positive_prob = float(p[0, positive_idx[0]])

        if model_type == "lgbm":
            if lgbm_prob is not None:
                raise ValueError("Multiple LightGBM models found.")
            lgbm_prob = positive_prob

        elif model_type == "rf":
            if rf_prob is not None:
                raise ValueError("Multiple Random Forest models found.")
            rf_prob = positive_prob

    # 두 모델 모두 확인
    if lgbm_prob is None:
        raise ValueError("LightGBM model was not found.")

    if rf_prob is None:
        raise ValueError("Random Forest model was not found.")

    # Fixed weighted ensemble: 80% + 20%
    final_prob = (
        0.8 * lgbm_prob
        + 0.2 * rf_prob
    )

    return np.array([
        [1.0 - final_prob, final_prob]
    ])


class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return weighted_predict_proba(self, X)

    def predict(self, X):
        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)
    
# Page settings

st.set_page_config(
    page_title="Adolescent Bullying Prediction Using Behavioral Characteristics",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .app-subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.4rem;
    }
    .result-number {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0 0.4rem 0;
    }
    .result-label {
        color: #5f6368;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Adolescent Bullying Prediction Using Behavioral Characteristics")
st.markdown(
    '<div class="app-subtitle">A research-based tool for predicting bullying using behavioral characteristics.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Before you begin")
    st.write(
        "This tool uses machine learning models to estimate the likelihood of bullying among adolescents based on their behavioral characteristics."
        "It is used for research and educational purposes and should not be interpreted as a definitive assessment of individual bullying risk."
    )

# Load final ensemble LightGBM+RF model and feature order

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ensemble_lgbm_rf.joblib"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"



@st.cache_resource
def load_assets():

    bundle = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_PATH)

    if isinstance(bundle, dict):

        model = bundle.get("model", bundle)
        threshold = bundle.get("threshold", None)

    else:

        model = bundle
        threshold = None

    feature_names = [str(x) for x in feature_names]

    # 앙상블 구조 및 개별 모델 확인
    base_models = extract_base_models(model)

    model_types = [
        identify_model_type(name, est)
        for name, est in base_models
    ]

    if model_types.count("lgbm") != 1:
        raise ValueError(
            "Exactly one LightGBM model is required."
        )

    if model_types.count("rf") != 1:
        raise ValueError(
            "Exactly one Random Forest model is required."
        )

    # Feature dimension 확인
    expected_n = getattr(model, "n_features_in_", None)

    if expected_n is not None and len(feature_names) != expected_n:

        raise ValueError(
            f"Feature mismatch: model expects {expected_n} "
            f"features but feature_names.pkl contains "
            f"{len(feature_names)}."
        )

    return model, threshold, feature_names


try:
    model, threshold, feature_names = load_assets()
except Exception as e:
    st.error(f"Failed to load model assets: {e}")
    st.stop()

# User inputs

st.markdown("### Enter information")

sex_label = st.selectbox("Sex", ["Male", "Female"])
Sex = {"Male": 1, "Female": 2}[sex_label]

grade_label = st.selectbox("Grade", ["Middle school (7th-9th)", "High school (10th-12th)"])
Grade = {"Middle school (7th-9th)": 1, "High school (10th-12th)": 2}[grade_label]

region_label = st.selectbox("Region of residence", ["Urban", "Rural"])
Region = {"Urban": 1, "Rural": 2}[region_label]

academic_label = st.selectbox("Academic achievement", ["Low", "Lower-middle", "Middle", "Upper-middle", "High"])
Academic_achievement = {"Low": 1, "Lower-middle": 2, "Middle": 3, "Upper-middle": 4, "High": 5}[academic_label]

income_label = st.selectbox("Household income", ["Low", "Lower-middle", "Middle", "Upper-middle", "High"])
Household_income = {"Low": 1, "Lower-middle": 2, "Middle": 3, "Upper-middle": 4, "High": 5}[income_label]

bmi_label = st.selectbox("BMI group", ["Underweight", "Normal", "Overweight", "Obese"])
BMI = {"Underweight": 1, "Normal": 2, "Overweight": 3, "Obese": 4,}[bmi_label]

sadness_label = st.selectbox("Sadness and despair", ["No", "Yes"])
Sadness_and_despair = {"No": 0, "Yes": 1}[sadness_label]

body_label = st.selectbox("Subjective body shape perception", ["Thin", "Normal", "Obese"])
Subjective_body_shape_perception = {"Thin": 1, "Normal": 2, "Obese": 3}[body_label]

smoking_label = st.selectbox("Smoking status", ["Nonsmoker", "Smoker"])
Smoking_status = {"Nonsmoker": 0, "Smoker": 1}[smoking_label]

alcohol_label = st.selectbox("Alcohol consumption",["Nondrinker", "More than one time"])
Alcohol_consumption = {"Nondrinker": 0, "More than one time": 1}[alcohol_label]

screentime_label = st.selectbox("Screentime status",["Low", "High"],)
Screentime_status = {"Low": 0, "High": 1}[screentime_label]

pa_label = st.selectbox("Physical activity", ["Infrequent", "Adequate"])
Physical_activity = {"Infrequent": 0, "Adequate": 1}[pa_label]

sleep_label = st.selectbox("Recommended_sleep_duration", ["No", "Yes"])
Recommended_sleep_duration = {"No": 0, "Yes": 1}[sleep_label]


# Reproduce the one-hot encoding used for model training

raw_values = {
    "Sex": Sex,
    "Grade": Grade,
    "Region": Region,
    "BMI": BMI,
    "Academic_achievement": Academic_achievement,
    "Household_income": Household_income,
    "Sadness_and_despair": Sadness_and_despair,
    "Self_weight": Subjective_body_shape_perception,
    "Smoking_status": Smoking_status,
    "Alcohol_consumption": Alcohol_consumption,
    "Screentime_status": Screentime_status,
    "Exercise_status": Physical_activity,
    "Sleep_duration": Recommended_sleep_duration
}

def encode_input(values, ordered_features):
    encoded = {name: 0.0 for name in ordered_features}

    categorical_vars = [
        "Sex",
        "Grade",
        "Region",
        "BMI",
        "Academic_achievement",
        "Household_income",
        "Sadness_and_despair",
        "Self_weight",
        "Smoking_status",
        "Alcohol_consumption",
        "Screentime_status",
        "Exercise_status",
        "Sleep_duration"
    ]

    for variable in categorical_vars:
        column = f"{variable}_{values[variable]}"
        if column not in encoded:
            raise ValueError(f"Encoded feature '{column}' is not present in feature_names.pkl.")
        encoded[column] = 1.0

    # The fitted model was trained on a 32-column one-hot encoded matrix.
    return np.array([[encoded[name] for name in ordered_features]], dtype=float)


# Prediction


# =========================================================
# Prediction
# =========================================================

if st.button(
    "Predict probability",
    type="primary",
    use_container_width=True
):

    try:

        X_input = encode_input(
            raw_values,
            feature_names
        )

        # LightGBM 80% + RF 20%
        probabilities = weighted_predict_proba(
            model,
            X_input
        )

        prob = float(probabilities[0, 1])

        st.markdown("---")

        st.markdown("## Prediction result")

        with st.container(border=True):

            st.markdown(
                f"""
                <div class="result-label">
                    Model-estimated probability
                </div>

                <div class="result-number">
                    {prob:.1%}
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.container(border=True):

            st.markdown("#### What this means")

            st.write(
                "This percentage represents the model-estimated "
                "probability of bullying victimization among adolescents "
                "based on the information entered above. "
                "A higher percentage indicates a higher predicted probability."
            )

        with st.container(border=True):

            st.markdown("#### Keep in mind")

            st.write(
                "This estimate reflects model predictions and may not "
                "represent an individual's actual experience. "
                "It should be used for research and educational purposes "
                "and should not be interpreted as a definitive assessment "
                "of individual bullying risk."
            )

        with st.expander("View information used for this prediction"):

            display_df = pd.DataFrame([
                {
                    "Sex": sex_label,
                    "Grade": grade_label,
                    "Region of residence": region_label,
                    "Academic achievement": academic_label,
                    "Household income": income_label,
                    "BMI group": bmi_label,
                    "Sadness and despair": sadness_label,
                    "Subjective body shape perception": body_label,
                    "Smoking status": smoking_label,
                    "Alcohol consumption": alcohol_label,
                    "Screentime status": screentime_label,
                    "Physical activity": pa_label,
                    "Recommended sleep duration": sleep_label
                }
            ])

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        with st.expander("View model configuration"):

            st.write("Ensemble model: LightGBM + Random Forest")

            st.write("LightGBM weight: 80%")

            st.write("Random Forest weight: 20%")

            st.write(
                f"LightGBM predicted probability: "
                f"{probabilities[0, 1]:.4f}"
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
