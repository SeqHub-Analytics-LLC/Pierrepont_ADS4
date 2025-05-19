import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import CATEGORICAL_COLS, NUMERICAL_COLS, BMI_BINS, BMI_LABELS
from model_utils import save_pickle

def feature_engineering(df):
    df['Player_Weight'] = df['Player_Weight'].round(2)
    df['Player_Height'] = df['Player_Height'].round(2)
    df['Training_Intensity'] = df['Training_Intensity'].round(2)

    # Add BMI column and BMI classification
    df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
    df['BMI_Classification'] = pd.cut(df['BMI'], bins=BMI_BINS, labels=BMI_LABELS, right=False)

    # Create Age Group column
    age_bins = [18, 22, 26, 30, 34, df['Player_Age'].max()]
    age_labels = ["18-22", "23-26", "27-30", "31-34", "35+"]
    df['Age_Group'] = pd.cut(df['Player_Age'], bins=age_bins, labels=age_labels, include_lowest=True)

    return df

def split_data(df):
    X = df.drop(columns=['Likelihood_of_Injury'])
    y = df['Likelihood_of_Injury']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def encode_features(X_train, X_test):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])

    X_train_encoded = encoder.transform(X_train[CATEGORICAL_COLS])
    X_test_encoded = encoder.transform(X_test[CATEGORICAL_COLS])

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLS)

    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

    X_train = X_train.drop(columns=CATEGORICAL_COLS)
    X_test = X_test.drop(columns=CATEGORICAL_COLS)

    X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

    save_pickle(encoder, 'pickles/encoder.pkl')

    return X_train, X_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERICAL_COLS])

    X_train_scaled = scaler.transform(X_train[NUMERICAL_COLS])
    X_test_scaled = scaler.transform(X_test[NUMERICAL_COLS])

    X_train.loc[:, NUMERICAL_COLS] = X_train_scaled
    X_test.loc[:, NUMERICAL_COLS] = X_test_scaled

    save_pickle(scaler, 'pickles/standard_scaler.pkl')

    return X_train, X_test