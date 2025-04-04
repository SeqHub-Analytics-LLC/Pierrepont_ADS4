df['Player_Weight'] = df['Player_Weight'].round(2)
df['Player_Height'] = df['Player_Height'].round(2)
df['Training_Intensity'] = df['Training_Intensity'].round(2)

# Feature Engineering

# Calculate the Body Mass Index (BMI)
df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2

# Defining gaps for BMI classification
gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']

# Create "BMI_Classification" column | Criar a coluna "Classificação_IMC"
df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)

df.head(1)

# Creating columns with grouping
df["Age_Group"] = pd.cut(
    df["Player_Age"],
    bins=[18, 22, 26, 30, 34, df["Player_Age"].max()],
    labels=["18-22", "23-26", "27-30", "31-34", "35+"],
    include_lowest=True,
)


# Encoding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Step 1: Split your data first
X = df.drop(columns=["Likelihood_of_Injury"])  # Features
y = df["Likelihood_of_Injury"]  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Select categorical columns to encode
one_hot_cols = ["BMI_Classification", "Age_Group","Training_Surface", "Position"]

# Step 3: Apply OneHotEncoder on training categorical columns only
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[one_hot_cols])

# Step 4: Apply the same encoder to transform the test set
X_test_encoded = encoder.transform(X_test[one_hot_cols])

# Step 5: Get encoded feature names
encoded_feature_names = encoder.get_feature_names_out(one_hot_cols)

# Step 6: Convert to DataFrames
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

# Step 7: Drop original categorical columns from X
X_train_clean = X_train.drop(columns=one_hot_cols)
X_test_clean = X_test.drop(columns=one_hot_cols)

# Step 8: Concatenate encoded columns with remaining features
X_train_final = pd.concat([X_train_clean, X_train_encoded_df], axis=1)
X_test_final = pd.concat([X_test_clean, X_test_encoded_df], axis=1)


# Scaling 

from sklearn.preprocessing import StandardScaler
import pickle

# Features to scale
scale_cols = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries',
              'Training_Intensity', 'Recovery_Time']

# Initialize scaler
scaler = StandardScaler()

# Fit on training data, transform both train and test
X_train_scaled = scaler.fit_transform(X_train_final[scale_cols])
X_test_scaled = scaler.transform(X_test_final[scale_cols])

# Replace scaled columns in X_train_final and X_test_final
X_train_final[scale_cols] = X_train_scaled
X_test_final[scale_cols] = X_test_scaled

# Save the scaler as a pickle file
with open("standard_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)