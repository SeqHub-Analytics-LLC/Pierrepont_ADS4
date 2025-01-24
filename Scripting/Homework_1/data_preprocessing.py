import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def clean_squat_data(df):
    def convert(x):
        try:
            x = float(x)
        except:
            return x
        return x

    invalid_squat_data = [*map(lambda x: type(convert(x)) != float, df['BestSquatKg'].unique())]
    wrong_data = df['BestSquatKg'].unique()[invalid_squat_data]
    corrected_data = np.array([*map(lambda x: x[:-2] + x[-1], wrong_data)]).astype('float')

    for i in range(len(wrong_data)):
        df.loc[df['BestSquatKg'] == wrong_data[i], 'BestSquatKg'] = corrected_data[i]

    df['BestSquatKg'] = df['BestSquatKg'].astype('float')
    return df

def handle_missing_values(train, test, column):
    mean_value = train[column].mean()
    train[column].fillna(mean_value, inplace=True)
    test[column].fillna(mean_value, inplace=True)
    return train, test

def preprocess_input_data(input_features):
    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])
    input_features.drop(columns=["Name", "playerId"], inplace=True)
    return input_features

def preprocess_target_data(targets):
    targets.drop(columns=["playerId"], inplace=True)
    return targets

def create_age_groups(df):
    age_bins = [0, 18, 23, 38, 49, 59, 69, float('inf')]
    age_labels = ['18 and under', '19-23', '24-38', '39-49', '50-59', '60-69', '70+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    return df, age_labels

def encode_age_groups(df, labels):
    ordinal_encoder = OrdinalEncoder(categories=[labels])
    df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(df[['AgeCategory']])
    return df

def add_relative_strengths(df):
    df['RelativeSquatStrength'] = df['BestSquatKg'] / df['BodyweightKg']
    df['RelativeDeadliftStrength'] = df['BestDeadliftKg'] / df['BodyweightKg']
    return df

def encode_categorical_data(df):
    equipment_scores = {
        'Raw': 1,
        'Wraps': 2,
        'Single-ply': 3,
        'Multi-ply': 4
    }
    df['Equipment_Index'] = df['Equipment'].map(equipment_scores)
    df = pd.get_dummies(df, columns=['Equipment', "Sex"], drop_first=True)
    return df

def split_and_scale_data(input_features, target, train_size):
    X_train = input_features.iloc[:train_size, :]
    X_test = input_features.iloc[train_size:, :]
    y_train = target.iloc[:train_size]
    y_test = target.iloc[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    return X_train_scaled, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv('processed_data/X_train.csv', index=False)
    X_test.to_csv('processed_data/X_test.csv', index=False)
    y_train.to_csv('processed_data/y_train.csv', index=False)
    y_test.to_csv('processed_data/y_test.csv', index=False)

def data_preprocessing(paths):
    data = pd.read_csv(paths)

    data = clean_squat_data(X)
    handle_missing_values(X, X, 'Age')

    input_features = preprocess_input_data(X)
    targets = preprocess_target_data(y)

    input_features = create_age_groups(input_features, 'Age')
    input_features = add_relative_strengths(input_features)
    input_features = encode_categorical_data(input_features)

    target = targets['BestBenchKg']
    input_features.drop(columns=['BestBenchKg', 'Age'], inplace=True)

    train_size = int(len(input_features) * 0.8)
    X_train_scaled, X_test, y_train, y_test = split_and_scale_data(input_features, target, train_size)

    save_data(X_train_scaled, X_test, y_train, y_test)

    return input_features, X_train_scaled, X_test, y_train, y_test

if __name__ == "__main__":
    paths = 'data/Power_lifting.csv'
    _,X_train_scaled,_,_= data_preprocessing(paths)
    print(X_train_scaled.head())


    