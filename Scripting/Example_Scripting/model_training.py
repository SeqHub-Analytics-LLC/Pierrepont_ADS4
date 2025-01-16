import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import os
import joblib

def objective(trial, X_train, y_train):
    """Objective function for Optuna to optimize Random Forest hyperparameters."""
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # Create the Random Forest model with suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Use cross-validation to evaluate the model
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False)
    )
    return -scores.mean()  # Return negative MSE for minimization

def train_model_with_optuna(X_train, y_train, n_trials=50):
    """Trains a Random Forest model using Optuna for hyperparameter tuning."""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    print(f"Best trial parameters: {study.best_params}")
    print(f"Best trial score: {study.best_value}")
    
    # Train the final model with the best parameters
    best_params = study.best_params
    best_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    return best_model, study

def save_model(model, output_path):
    """Saves the trained model."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    joblib.dump(model, output_path)

