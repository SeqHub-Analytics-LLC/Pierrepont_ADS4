# Constants for columns
one_hot_cols = ["BMI_Classification", "Age_Group"]
scale_cols = ["Player_Age", "Player_Weight", "Player_Height", "Previous_Injuries", "Training_Intensity", "Recovery_Time"]

# Paths
ENCODER_PATH = "pickles/one_hot_encoder.pkl"
SCALER_PATH = "pickles/standard_scaler.pkl"
MODEL_PATH = "pickles/random_forest_model.pkl"
