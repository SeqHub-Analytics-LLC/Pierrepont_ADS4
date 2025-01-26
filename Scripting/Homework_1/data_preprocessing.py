import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

def train_split_save_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    return train, test

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

def preprocess_data(train):
    kg_features = train.filter(regex='Kg').columns
    train[kg_features] = np.abs(train[kg_features])
    train.drop(columns=["Name", "playerId"], inplace=True)
    return train

def create_age_groups(df):
    age_bins = [0, 18, 23, 38, 49, 59, 69, float('inf')]
    age_labels = ['18 and under', '19-23', '24-38', '39-49', '50-59', '60-69', '70+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df.drop(columns=['Age'], inplace=True)
    return df, age_labels

def encode_age_groups(df, labels):
    ordinal_encoder = OrdinalEncoder(categories=[labels])
    df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(df[['AgeGroup']])
    df.drop(columns=['AgeGroup'], inplace=True)
    return df

def add_relative_strengths(df):
    df['RelativeSquatStrength'] = df['BestSquatKg'] / df['BodyweightKg']
    df['RelativeDeadliftStrength'] = df['BestDeadliftKg'] / df['BodyweightKg']
    df = df.drop(columns=['BestSquatKg', 'BestDeadliftKg'])
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

def scale_split_data(train, test, target):
    X_train = train
    y_train = train[target]
    X_test = test
    y_test = test[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    return X_train_scaled, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv('processed_data\X_train_scaled.csv', index=False)
    X_test.to_csv('processed_data\X_test.csv', index=False)
    y_train.to_csv('processed_data\y_train.csv', index=False)
    y_test.to_csv('processed_data\y_test.csv', index=False)

def data_preprocessing(path):
    data = pd.read_csv(path)

    train, test = train_split_save_data(data)
    train, test = clean_squat_data(train), clean_squat_data(test)
    train, test = handle_missing_values(train, test, 'Age')

    train, test = preprocess_data(train), preprocess_data(test)  

    train, train_age_labels = create_age_groups(train)
    test, test_age_labels = create_age_groups(test)

    train = encode_age_groups(train, train_age_labels)
    test = encode_age_groups(test, test_age_labels)

    train = add_relative_strengths(train)
    test = add_relative_strengths(test)
    
    train = encode_categorical_data(train)
    test = encode_categorical_data(test)

    target = 'BestBenchKg'

    X_train_scaled, X_test, y_train, y_test = scale_split_data(train, test, target)

    save_data(X_train_scaled, X_test, y_train, y_test)

    return X_train_scaled, X_test, y_train, y_test

if __name__ == "__main__":
    os.chdir("Scripting\Homework_1")
    paths = 'data\Power_lifting.csv'
    data = pd.read_csv(paths)
    train, test = train_split_save_data(data)
    data = clean_squat_data(data)


    