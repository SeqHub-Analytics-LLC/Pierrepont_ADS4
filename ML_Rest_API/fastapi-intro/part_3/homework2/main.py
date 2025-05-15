import pandas as pd
from preprocessing import (
    feature_engineering,
    split_data,
    encode_features,
    scale_features
)
from train_model import run_optimization, evaluate_model
from config import DATA_PATH

def main():
    df = pd.read_csv(DATA_PATH)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = encode_features(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)
    model = run_optimization(X_train, y_train)
    evaluate_model(model, X_test, y_test)

=======
from processing import load_and_clean_data, scale_features, save_preprocessed_data, encode_features, create_features
from train_model import train_model_with_optuna, save_model
from eval import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Step 1: Load and preprocess data
    data = load_and_clean_data("data/injury_data_with_categories.csv")
    X = data.drop(columns=['Likelihood_of_Injury'])
    y = data['Likelihood_of_Injury']
    
    X = create_features(X)
    X, _ = encode_features(X, use_saved=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save preprocessed data
    save_preprocessed_data(pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test), "data/preprocessed_data")
    
    #scale the data
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, use_saved=False)
   
    # Train the model using Optuna
    best_model, study = train_model_with_optuna(X_train_scaled, y_train, n_trials=50)
    
    # Save the model and study
    save_model(best_model, "data/artifacts/random_forest_model.pkl") # Save the model

    #Evaluate the model
    accuracy, auc = evaluate_model(best_model, X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy}, AUC Score: {auc}")

if __name__ == "__main__":
    main()
