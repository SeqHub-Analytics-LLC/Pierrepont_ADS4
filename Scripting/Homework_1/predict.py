import joblib
import pandas as pd
import numpy as np
from data_processing import load_and_clean_data, fix_best_squat,engineer_features

def make_prediction(model_path, test_file_path):
    """Loads a saved model and evaluates it on new test data."""
    model = joblib.load(model_path)
    data = pd.read_csv(test_file_path)
    clean_data = fix_best_squat(data)
    input_features = clean_data.drop(columns =["playerId","Name"])
    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])
    input_features = engineer_features(input_features)

    #loading the saved encoders
    oh_encoder = joblib.load("model_output/oh_enocoder.pkl")
    ordinal_encoder = joblib.load("model_output/ordinal_encoder.pkl")

    
    input_encoded = oh_encoder.transform(input_features[['Equipment','Sex']])

    #GET COLUMN NAMES FOR NEWLY ENCODED VARIABLES
    encoded_columns = oh_encoder.get_feature_names_out(input_features = ['Equipment','Sex'])

    #only encoded variables
    encoded_variables = pd.DataFrame(input_encoded,columns=encoded_columns)

    #combine with other input variables
    full_data = pd.concat([input_features,encoded_variables], axis=1)

    ## Bin Age
    # Define the bins for the age categories
    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

    # Bin the age data
    full_data['AgeCategory'] = pd.cut(full_data['Age'], bins=bins, labels=labels, right=False)
    #Encode Bins
    full_data['AgeCategoryEncoded'] = ordinal_encoder.transform(full_data[['AgeCategory']])
    
    #Drop redundant features
    full_data.drop(columns = ["Sex", "Equipment", "Age","AgeCategory"],inplace=True)

    predictions = model.predict(full_data)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    make_prediction("model_output/random_forest_model.pkl", "test_data.csv")
 