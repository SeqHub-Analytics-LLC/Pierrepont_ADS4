import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

from utils import engineer_features, encode_features, scale_features 

def train_and_save_model():
    df = pd.read_csv('injury_data_with_categories.csv')
    X = df.drop(columns=['Likelihood_of_Injury'])
    y = df['Likelihood_of_Injury']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    categorical_cols = ["Position", "Training_Surface"]
    X_train = encode_features(X_train, categorical_cols, path="artifacts/oneHotEncoder.pkl")
    X_test = encode_features(X_test, categorical_cols, path="artifacts/oneHotEncoder.pkl")
    numeric_cols = ["Player_Weight", "Player_Height", "Player_Age", "Training_Intensity", "Recovery_Time"]
    X_train = scale_features(X_train, numeric_cols, path="artifacts/standardScaler.pkl")
    X_test = scale_features(X_test, numeric_cols, path="artifacts/standardScaler.pkl")
    rf.fit(X_train, y_train)
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(rf, 'artifacts/random_forest_model.pkl')
    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, 'artifacts/model_features.pkl')

if __name__ == "__main__":
    train_and_save_model()
