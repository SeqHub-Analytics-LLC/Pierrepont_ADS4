import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Train your Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_final, y_train)

# 2. Save the trained model as a .pkl file
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)


with open("random_forest_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make predictions
preds = loaded_model.predict(new_samples)


with open("standard_scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

# Then use it like this:
scaled_data = loaded_scaler.transform(new_data[scale_cols])


with open("one_hot_encoder.pkl", "rb") as f:
    loaded_encoder = pickle.load(f)

# Then use it like this:
X_new_encoded = loaded_encoder.transform(new_data[one_hot_cols])
