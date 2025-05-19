import streamlit as st
import requests

st.title("Sports Injury ML Model Predictor")
st.write("Enter player details to predict the likelihood of injury.")

# Input form
def input_form():
    with st.form("prediction_form"):
        Player_Weight = st.number_input("Player Weight (kg)", min_value=30.0, max_value=150.0, value=75.0)
        Player_Height = st.number_input("Player Height (cm)", min_value=120.0, max_value=220.0, value=180.0)
        Previous_Injuries = st.selectbox("Previous Injuries", ["Yes", "No"])
        Position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
        Training_Surface = st.selectbox("Training Surface", ["Grass", "Artificial Turf", "Hard Court"])
        Player_Age = st.number_input("Player Age", min_value=10, max_value=50, value=25)
        Training_Intensity = st.slider("Training Intensity", min_value=0.0, max_value=1.0, value=0.8)
        Recovery_Time = st.number_input("Recovery Time (days)", min_value=0.0, max_value=30.0, value=1.5)
        submitted = st.form_submit_button("Predict")
    return submitted, {
        "Player_Weight": Player_Weight,
        "Player_Height": Player_Height,
        "Previous_Injuries": Previous_Injuries,
        "Position": Position,
        "Training_Surface": Training_Surface,
        "Player_Age": Player_Age,
        "Training_Intensity": Training_Intensity,
        "Recovery_Time": Recovery_Time
    }

submitted, features = input_form()

if submitted:
    payload = {
        "Features": features,
        "model": "Random Forests"
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Likelihood of Injury: {result['prediction']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# Batch prediction UI
st.header("Batch Prediction")
st.write("Upload a CSV file with player data for batch predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    if st.button("Predict Batch"):
        batch_payload = [
            {"Features": row, "model": "Random Forests"}
            for row in df.to_dict(orient="records")
        ]
        try:
            response = requests.post("http://localhost:8000/predict_batch", json=batch_payload)
            if response.status_code == 200:
                result = response.json()
                st.success("Batch predictions complete!")
                st.write(result["predictions"])
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
