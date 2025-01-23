from model_training import train_model_with_optuna, tune_decision_tree
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing import load_and_clean_data
from utils import save_preprocessed_data, save_decision_tree_feature_importance, save_model
from model_validation import evaluate_model

def main(model_type):
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    inputs, targets = load_and_clean_data("Power_lifting.csv")

    #No need to scale - Tree Based Models
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.25, random_state=42)

    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, "preprocessed_data")
    
    if model_type == "Random Forests":
        # Step 2: Train the model using Optuna
        best_model, study = train_model_with_optuna(X_train, y_train, n_trials=50)
        
        # Save the model and study
        save_model(best_model, "model_output/random_forest_model.pkl")
        joblib.dump(study, "model_output/optuna_study.pkl")  # Save the Optuna study for analysis
        #feature importance plot
        save_decision_tree_feature_importance(best_model, X_train.columns, "model_output/Random_Forest_FI_plot")
        # Step 3: Evaluate the model
        rf_metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
        print(rf_metrics)

    elif model_type == "Decision Trees":
        # Step 2: Train the model using Grid Search
        best_model = tune_decision_tree(X_train, y_train)
        
        # Save the model and study
        save_model(best_model, "model_output/dtc_model.pkl")
        #feature importance plot
        save_decision_tree_feature_importance(best_model, X_train.columns, "model_output/DTC_FI_plot")
        # Step 3: Evaluate the model
        rf_metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
        print(rf_metrics)
    else:
        print("Unrecognized Model Name.")


if __name__ == "__main__":
    main(model_type="Random Forests")