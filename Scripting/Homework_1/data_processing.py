
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def load_data():
    data=pd.read_csv("Power_lifting.csv")
    
    return data

def load_datasets(train_path, test_path, train_target_path, test_target_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(train_target_path)
    y_test = pd.read_csv(test_target_path)
    return X_train, X_test, y_train, y_test

def clean_y_test(y_test):
    y_test = y_test.drop(columns=["Age", "BodyweightKg", "BestDeadliftKg"], inplace=False)
    return y_test

def convert_to_float(x):
    try:
        x = float(x)
    except:
        return x
    return x

def correct_best_squat_kg(X_train, X_test):
    def fix_column(data, column_name):
        wrong_data = data[column_name].unique()[[*map(lambda x: type(convert_to_float(x)) != float, data[column_name].unique())]]
        corrected_data = np.array([*map(lambda x: x[:-2] + x[-1], wrong_data)]).astype('float')
        for i in range(len(wrong_data)):
            data.loc[data[column_name] == wrong_data[i], column_name] = corrected_data[i]
        data[column_name] = data[column_name].astype('float')

    fix_column(X_train, 'BestSquatKg')
    fix_column(X_test, 'BestSquatKg')
    return X_train, X_test

def fill_missing_ages(X_train, X_test):

    mean_age = X_train['Age'].mean()
    X_train['Age'].fillna(mean_age, inplace=True)
    X_test['Age'].fillna(mean_age, inplace=True)
    return X_train, X_test

def preprocess_features_targets(X_train, X_test, y_train, y_test):
    input_features = pd.concat([X_train, X_test], axis=0)
    targets = pd.concat([y_train, y_test], axis=0)

    # Ensure kg features are positive
    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])

    # Drop unnecessary columns
    input_features.drop(columns=["Name", "playerId"], inplace=True)
    targets.drop(columns=["playerId"], inplace=True)

    return input_features, targets

def calculate_relative_strength(data_df):
    data_df['RelativeSquatStrength'] = data_df['BestSquatKg'] / data_df['BodyweightKg']
    data_df['RelativeDeadliftStrength'] = data_df['BestDeadliftKg'] / data_df['BodyweightKg']
    return data_df

def map_equipment(data_df):
    """Map equipment types to their respective scores."""
    equipment_scores = {
        'Raw': 1,
        'Wraps': 2,
        'Single-ply': 3,
        'Multi-ply': 4
    }
    data_df['Equipment_Index'] = data_df['Equipment'].map(equipment_scores)
    return data_df

def encode_age_category(data_df):
    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

    data_df['AgeCategory'] = pd.cut(data_df['Age'], bins=bins, labels=labels, right=False)

    ordinal_encoder = OrdinalEncoder(categories=[labels])
    data_df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(data_df[['AgeCategory']])
    return data_df

def preprocess_pipeline(train_path, test_path, train_target_path, test_target_path):
    X_train, X_test, y_train, y_test = load_datasets(train_path, test_path, train_target_path, test_target_path)

    y_test = clean_y_test(y_test)
    X_train, X_test = correct_best_squat_kg(X_train, X_test)
    X_train, X_test = fill_missing_ages(X_train, X_test)

    input_features, targets = preprocess_features_targets(X_train, X_test, y_train, y_test)
    data_df = pd.concat([input_features, targets], axis=1)

    data_df = calculate_relative_strength(data_df)
    data_df = map_equipment(data_df)
    data_df = encode_age_category(data_df)

    return data_df
if __name__=="__main__":
    correct_best_squat_kg(X_train,X_test)
