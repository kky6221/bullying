from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin

class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    # 문제가 되던 __getattr__ 메서드는 완전히 삭제합니다. (scikit-learn 기본 동작을 따르게 함)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # 1. 모델 추출 (sklearn 호환성 보장)
        estimators = getattr(self, "estimators_", None)
        if estimators is None:
            estimators = getattr(self, "estimators", [])
        
        # 튜플 형태(이름, 모델)로 저장된 경우 모델(객체)만 추출
        if estimators and isinstance(estimators[0], tuple):
            estimators = [est for _, est in estimators]
            
        if not estimators:
            raise ValueError("앙상블에 포함된 예측 모델(estimators)이 없습니다.")

        # 2. 모든 개별 모델의 예측 확률 계산
        probas = np.array([est.predict_proba(X) for est in estimators])

        # 3. 가중치 안전하게 추출
        weights = getattr(self, "weights", None)

        # 4. 에러 원천 차단: 가중치가 None이거나, 모델 개수와 다르거나, 합이 0인 경우
        if weights is None or len(weights) != len(estimators) or np.sum(weights) == 0:
            # 0으로 나누는 에러를 방지하기 위해 가중치 계산을 생략하고 단순 평균(np.mean) 사용
            return np.mean(probas, axis=0)

        # 정상적인 가중치가 존재하는 경우 (np.average가 내부적으로 가중합을 나눠줍니다)
        return np.average(probas, axis=0, weights=weights)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
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

    # bundle이 딕셔너리 형태로 저장된 경우와 모델 객체로 직접 저장된 경우 모두 처리
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
        threshold = bundle.get("threshold", None)
    else:
        model = bundle
        threshold = None

    feature_names = [str(x) for x in feature_names]

    if len(feature_names) != getattr(
        model, "n_features_in_", len(feature_names)
    ):
        raise ValueError(
            f"Feature mismatch: model expects {getattr(model, 'n_features_in_', 'unknown')} "
            f"features but feature_names.pkl contains {len(feature_names)}."
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

if st.button("Predict probability", type="primary", use_container_width=True):
    try:
        X_input = encode_input(raw_values, feature_names)
        prob = float(model.predict_proba(X_input)[0, 1])

        st.markdown("---")
        st.markdown("## Prediction result")

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-label">Model-estimated probability</div>
                <div class="result-number">{prob:.1%}</div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### What this means")
            st.write(
                "This percentage is the model-estimated probability of behavioral character-related bullying"
                "based on the information entered above. A higher percentage indicates a higher predicted probability."
            )

        with st.container(border=True):
            st.markdown("#### Keep in mind")
            st.write(
                "This estimate reflects model predictions and may not represent an individual bullying risk. "
                "It is intended for research and educational use."
            )

        with st.expander("View information used for this prediction"):
            display_df = pd.DataFrame(
                [
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
                ]
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
