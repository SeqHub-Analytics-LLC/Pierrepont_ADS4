from data_processing import load_and_clean_data, scale_features, save_preprocessed_data
from model_training import train_model_with_optuna, save_model
from model_validation import evaluate_model
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Step 1: Load and preprocess data
    data = load_and_clean_data("Apple_demand.csv")
    X = data.drop(columns=['demand'])
    y = data['demand']
    
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Save preprocessed data
    save_preprocessed_data(pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test), "preprocessed_data")
    
    # Step 2: Train the model using Optuna
    best_model, study = train_model_with_optuna(X_train, y_train, n_trials=50)
    
    # Save the model and study
    save_model(best_model, "model_output/random_forest_model.pkl")
    joblib.dump(study, "model_output/optuna_study.pkl")  # Save the Optuna study for analysis
    
    # Step 3: Evaluate the model
    rmse, r2 = evaluate_model(best_model, X_test, y_test)
    print(f"Test RMSE: {rmse}, R2 Score: {r2}")

if __name__ == "__main__":
    main()
