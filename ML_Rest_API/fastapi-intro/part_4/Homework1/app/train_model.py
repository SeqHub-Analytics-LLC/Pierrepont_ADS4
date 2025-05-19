import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from model_utils import engineer_features, encode_features, scale_features  # Assumes these functions exist

# Load the data
df = pd.read_csv('injury_data_with_categories.csv')

# Split data

X = df.drop(columns=['Likelihood_of_Injury'])
y = df['Likelihood_of_Injury']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Engineer features
X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

# Encode categorical features
categorical_cols = ["Position", "Training_Surface"]
X_train = encode_features(X_train, categorical_cols)
X_test = encode_features(X_test, categorical_cols, path="artifacts/oneHotEncoder.pkl")

# Scale numeric features
numeric_cols = ["Player_Weight", "Player_Height", "Player_Age", "Training_Intensity", "Recovery_Time"]
X_train = scale_features(X_train, numeric_cols)
X_test = scale_features(X_test, numeric_cols, path="artifacts/standardScaler.pkl")


print(X_train.columns)
# Fit the model
#rf.fit(X_train, y_train)

# Save the model
#joblib.dump(rf, 'random_forest_model.joblib')

