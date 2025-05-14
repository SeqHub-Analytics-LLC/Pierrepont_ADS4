import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import os
import joblib

def objective(trial, X_train, y_train):
    """Objective function for Optuna to optimize Random Forest hyperparameters."""
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # define a stratified split
    skf = StratifiedKFold(n_splits=5,random_state=67, shuffle=True)

    # Create the Random Forest model with suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Use cross-validation to evaluate the model
    scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="roc_auc"
    )
    return scores.mean()

def train_model_with_optuna(X_train, y_train, n_trials=50):
    """Trains a Random Forest model using Optuna for hyperparameter tuning."""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    print(f"Best trial parameters: {study.best_params}")
    print(f"Best trial score: {study.best_value}")
    
    # Train the final model with the best parameters
    best_params = study.best_params
    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    return best_model, study

def save_model(model, output_path='data/artifacts'):
    """Saves the trained model."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    joblib.dump(model, output_path)

