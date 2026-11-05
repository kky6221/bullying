# Behavioral Character-Related Bullying Prediction

Streamlit application for estimating the probability of cost-related unmet healthcare needs using the final ensemble LightGBM+RF model.

## Required repository files

`app.py` — Streamlit application

`ensemble\\\_lgbm\\\_rf.joblib` — final fitted LightGBM model and stored threshold

`feature\\\_names.pkl` — exact 32-feature order used during model training

`requirements.txt` — Python package dependencies

`runtime.txt` — Python runtime for deployment

`README.md` is documentation only and is not required for the app to run.

## Model input

The app collects 13 predictors:

1. Sex
2. Grade
3. Region of residence
4. Academic achievement
5. Household income
6. BMI group
7. Sadness and despair
8. Subjective body shape perception
9. Smoking status
10. Alcohol consumption
11. Screentime status
12. Physical activity
13. Recommended sleep duration

Categorical inputs are converted to the same one-hot encoded 32-feature representation used for model training. Feature ordering is loaded directly from `feature\\\_names.pkl`.

## GitHub / Streamlit deployment

Upload all required files to the root of one GitHub repository. In Streamlit Community Cloud, select the repository and set the main file path to:

`app.py`

## Intended use

This application is intended for research and educational purposes. It is not a clinical diagnostic tool.

