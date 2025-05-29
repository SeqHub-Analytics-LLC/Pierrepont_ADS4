import streamlit as st
from PIL import Image
import requests
import pandas as pd
import io

st.title("Injury Prediction Dashboard")


prediction_type = st.sidebar.radio(
    "Choose Prediction Type",
    ["Single Prediction", "Batch Prediction"],
    index=None
)


if prediction_type is None:
    st.markdown("""
    **Welcome to the Injury Prediction System!**

    This application helps predict athlete injury risk based on training and physical metrics.
    """)

    try:
        image = Image.open("artifacts/injury_prediction.webp")
        st.image(image, caption="Injury Risk Assessment Tool", use_container_width=True)
    except FileNotFoundError:
        st.warning("Could not load the default image")

    st.markdown("""
    **How It Works:**
    - Provide athlete metrics through either single input or batch upload
    - Our machine learning model assesses injury risk
    - Get actionable insights to prevent injuries

    **Options:**
    - **Single Prediction:** Enter details for one athlete
    - **Batch Prediction:** Upload CSV file for multiple athletes
    """)

elif prediction_type == "Single Prediction":
    st.markdown("### Single Athlete Injury Prediction")
    st.markdown("Enter the athlete's details to assess injury risk:")
    st.markdown("#### Athlete Information")
    col1, col2 = st.columns(2)
    with col1:
        player_weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=75.0)
        player_height = st.number_input("Height (cm)", min_value=140.0, max_value=220.0, value=180.0)
        player_age = st.number_input("Age", min_value=16, max_value=50, value=25)
    
    with col2:
        previous_injuries = st.selectbox("Previous Injuries", ["No", "Yes"])
        position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
        training_surface = st.selectbox("Training Surface", ["Grass", "Artificial Turf", "Indoor"])

    st.markdown("#### Training Metrics")
    training_intensity = st.slider("Training Intensity (0-1)", 0.0, 1.0, 0.8)
    recovery_time = st.number_input("Recovery Time (hours)", min_value=0.0, value=1.5)

    if st.button("Predict Injury Risk"):
        payload = {
            "Features": {
                "Player_Weight": player_weight,
                "Player_Height": player_height,
                "Previous_Injuries": previous_injuries,
                "Position": position,
                "Training_Surface": training_surface,
                "Player_Age": player_age,
                "Training_Intensity": training_intensity,
                "Recovery_Time": recovery_time
            },
            "model": "Random Forests"
        }

        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", 0)                
            if prediction == 1:
                st.error("High injury risk detected! Consider adjusting training.")
            else:
                st.success("Low injury risk - training regimen appears safe.")
                
            prob_response = requests.post("http://127.0.0.1:8000/predict_probabilities", json=payload)
            if prob_response.status_code == 200:
                probs = prob_response.json()
                st.metric("Injury Probability", f"{probs['injured_prob']*100:.1f}%")
        else:
            st.error(f"Prediction failed: {response.text}")
 
elif prediction_type == "Batch Prediction":
    st.markdown("### Batch Injury Prediction")
    st.markdown("Upload a CSV file with athlete data to assess injury risk for multiple players.")

    st.markdown("""
    **Required CSV Columns:**
    - Player_Weight (kg)
    - Player_Height (cm)
    - Previous_Injuries (Yes/No)
    - Position
    - Training_Surface
    - Player_Age
    - Training_Intensity (0-1)
    - Recovery_Time (hours)
    """)

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())

            if st.button("Predict for All Athletes"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                
                with st.spinner("Processing predictions..."):
                    response = requests.post(
                        "http://127.0.0.1:8000/predict_batch",
                        files=files,
                        data={"model": "Random Forests"}
                    )

                if response.status_code == 200:
                    results = response.json()
                    predictions = results.get("predictions", [])
                    probabilities = results.get("probabilities", [])
                    df['Injury_Risk'] = ['High' if p == 1 else 'Low' for p in predictions]
                    df['Risk_Probability'] = [f"{prob[1]*100:.1f}%" for prob in probabilities]
                    st.success("Predictions completed!")
                    st.write("### Prediction Results")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        data=csv,
                        file_name="injury_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Prediction failed: {response.text}")

