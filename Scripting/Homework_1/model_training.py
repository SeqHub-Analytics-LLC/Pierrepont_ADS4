import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def objective(trial):
    X_train = pd.read_csv('processed_data/X_train.csv')
    y_train = pd.read_csv('processed_data/y_train.csv')

    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

    model = DecisionTreeRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    return rmse

def train_model_with_optuna(X_train, y_train, n_trials=50):
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Save the Optuna study
    study_path = "model_output/decision_tree_study.pkl"
    joblib.dump(study, study_path)

    # Train the best model using the best parameters
    best_params = study.best_params
    best_model = DecisionTreeRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Save the trained model
    model_path = "model_output/decision_tree_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {study.best_value}")
    print(f"Study saved at: {study_path}")
    print(f"Model saved at: {model_path}")

    return best_model, model_path, study_path