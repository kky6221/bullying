from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. 어떤 악조건에서도 모델을 강제로 찾아내고, 에러를 무시하는 완벽 방어 클래스
class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def __getattr__(self, name):
        # scikit-learn의 n_features_in_ 등 속성 검사 시 에러 없이 통과하도록 표준 방식을 따름
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        models = []
        
        # [핵심] 1. 객체 내부에 저장된 모든 리스트를 샅샅이 뒤져서 예측 모델을 강제로 찾아냅니다.
        for key, val in self.__dict__.items():
            if isinstance(val, (list, tuple)) and len(val) > 0:
                for item in val:
                    # 단일 모델 객체인 경우
                    if hasattr(item, 'predict_proba') and item is not self:
                        if item not in models:
                            models.append(item)
                    # (이름, 모델) 튜플 형태인 경우
                    elif isinstance(item, tuple) and len(item) >= 2:
                        if hasattr(item[1], 'predict_proba') and item[1] is not self:
                            if item[1] not in models:
                                models.append(item[1])
            if models:
                break # 모델을 하나라도 찾았으면 탐색 종료
                
        # [핵심] 2. 리스트가 아니라 개별 변수(예: self.model1, self.model2)로 저장된 경우도 대비
        if not models:
            for key, val in self.__dict__.items():
                if hasattr(val, 'predict_proba') and val is not self:
                    models.append(val)
                    
        # 3. 그래도 없으면 객체의 현재 상태를 화면에 출력하여 원인을 즉시 파악할 수 있게 함
        if not models:
            keys = list(self.__dict__.keys())
            raise ValueError(f"저장된 모델 파일 안에 예측 모델이 없습니다. 발견된 내부 속성들: {keys}")

        # 4. 각 모델별 예측 확률 계산
        probas = np.array([m.predict_proba(X) for m in models])
        
        # 5. 가중치(Weights) 0으로 나누기 에러 원천 차단
        try:
            w = self.weights
            if w is not None and len(w) == len(models) and sum(w) > 0:
                w_norm = np.array(w, dtype=float) / sum(w)
                return np.average(probas, axis=0, weights=w_norm)
        except Exception:
            pass # 가중치 계산 중 어떤 에러라도 발생하면 아래의 단순 평균으로 우회
            
        # 가중치에 문제가 있으면 가장 안전한 '단순 평균'으로 결과를 무조건 반환
        return np.mean(probas, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# Page settings
st.set_page_config(
    page_title="Adolescent Bullying risk Prediction Using Behavioral Characteristics",
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

st.title("Adolescent Bullying risk Prediction Using Behavioral Characteristics")
st.markdown(
    '<div class="app-subtitle">A research-based tool for predicting bullying risk using behavioral characteristics.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Before you begin")
    st.write(
        "This tool uses machine learning models to estimate the likelihood of bullying risk among adolescents based on their behavioral characteristics."
        " It is used for research and educational purposes and should not be interpreted as a definitive assessment of individual bullying risk."
    )

# Load final ensemble LightGBM+RF model and feature order
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ensemble_lgbm_rf.joblib"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"

# 2. [캐시 초기화용] 함수 이름을 load_assets_final로 변경하여 깃허브 배포 시 악성 캐시가 완전히 무효화되도록 처리
@st.cache_resource
def load_assets_final():
    bundle = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
        threshold = bundle.get("threshold", None)
    else:
        model = bundle
        threshold = None

    feature_names = [str(x) for x in feature_names]

    # getattr 기본값 동작 정상화
    if len(feature_names) != getattr(model, "n_features_in_", len(feature_names)):
        raise ValueError(
            f"Feature mismatch: model expects {getattr(model, 'n_features_in_', 'unknown')} "
            f"features but feature_names.pkl contains {len(feature_names)}."
        )

    return model, threshold, feature_names

try:
    model, threshold, feature_names = load_assets_final()
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
BMI = {"Underweight": 1, "Normal": 2, "Overweight": 3, "Obese": 4}[bmi_label]

sadness_label = st.selectbox("Sadness and despair", ["No", "Yes"])
Sadness_and_despair = {"No": 0, "Yes": 1}[sadness_label]

body_label = st.selectbox("Subjective body shape perception", ["Thin", "Normal", "Obese"])
Subjective_body_shape_perception = {"Thin": 1, "Normal": 2, "Obese": 3}[body_label]

smoking_label = st.selectbox("Smoking status", ["Non-smoker", "Smoker"])
Smoking_status = {"Non-smoker": 0, "Smoker": 1}[smoking_label]

alcohol_label = st.selectbox("Alcohol consumption",["Non-drinker", "More than one time"])
Alcohol_consumption = {"Non-drinker": 0, "More than one time": 1}[alcohol_label]

screentime_label = st.selectbox("Screentime status",["Low", "High"])
Screentime_status = {"Low": 0, "High": 1}[screentime_label]

pa_label = st.selectbox("Physical activity", ["Infrequent", "Adequate"])
Physical_activity = {"Infrequent": 0, "Adequate": 1}[pa_label]

sleep_label = st.selectbox("Recommended sleep duration", ["No", "Yes"])
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
        "Sex", "Grade", "Region", "BMI", "Academic_achievement",
        "Household_income", "Sadness_and_despair", "Self_weight",
        "Smoking_status", "Alcohol_consumption", "Screentime_status",
        "Exercise_status", "Sleep_duration"
    ]
    for variable in categorical_vars:
        column = f"{variable}_{values[variable]}"
        if column not in encoded:
            raise ValueError(f"Encoded feature '{column}' is not present in feature_names.pkl.")
        encoded[column] = 1.0
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
                "This percentage is the model-estimated probability of behavioral characteristics-related bullying risk "
                "based on the information entered above. A higher percentage indicates a higher predicted probability."
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
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
