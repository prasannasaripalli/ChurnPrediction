import os
import pickle

import mlflow
import mlflow.keras
import pandas as pd
import streamlit as st


def load_pickle_from_mlflow(uri):
    path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    with open(path, "rb") as f:
        return pickle.load(f)


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

run_id = os.getenv("MLFLOW_RUN_ID")
if not run_id:
    st.error("MLFLOW_RUN_ID is missing.")
    st.stop()

model_uri = f"runs:/{run_id}/model"
scaler_uri = f"runs:/{run_id}/scaler.pkl"
geo_uri = f"runs:/{run_id}/onehot_encoder_geo.pkl"
gender_uri = f"runs:/{run_id}/label_encoder_gender.pkl"


@st.cache_resource
def load_artifacts():
    model = mlflow.keras.load_model(model_uri)
    scaler = load_pickle_from_mlflow(scaler_uri)
    onehot_encoder_geo = load_pickle_from_mlflow(geo_uri)
    label_encoder_gender = load_pickle_from_mlflow(gender_uri)
    return model, scaler, onehot_encoder_geo, label_encoder_gender


try:
    model, scaler, onehot_encoder_geo, label_encoder_gender = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model or artifacts: {e}")
    st.stop()


def clear_form():
    st.session_state.geography = onehot_encoder_geo.categories_[0][0]
    st.session_state.gender = label_encoder_gender.classes_[0]
    st.session_state.age = 35
    st.session_state.balance = 0.0
    st.session_state.credit_score = 600
    st.session_state.estimated_salary = 50000.0
    st.session_state.tenure = 3
    st.session_state.num_of_products = 2
    st.session_state.has_cr_card = 1
    st.session_state.is_active_member = 1


st.title("Customer Churn Prediction")

# initialize defaults once, before widgets
if "geography" not in st.session_state:
    clear_form()

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0], key="geography")
gender = st.selectbox("Gender", label_encoder_gender.classes_, key="gender")
age = st.slider("Age", 18, 92, key="age")
balance = st.number_input("Balance", min_value=0.0, key="balance")
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, key="credit_score")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, key="estimated_salary")
tenure = st.slider("Tenure", 0, 10, key="tenure")
num_of_products = st.slider("Number of Products", 1, 4, key="num_of_products")
has_cr_card = st.selectbox("Has Credit Card", [0, 1], key="has_cr_card")
is_active_member = st.selectbox("Is Active Member", [0, 1], key="is_active_member")

col1, col2 = st.columns(2)
predict_clicked = col1.button("Predict", type="primary")
col2.button("Clear", on_click=clear_form)

if predict_clicked:
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    st.write(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.warning("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")