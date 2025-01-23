import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils import save_decision_tree_feature_importance


def objective(trial, X_train, y_train):
    """Objective function for Optuna to optimize Decision Tree hyperparameters."""
    # Define the hyperparameter search space
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # Create the Decision Tree model with suggested hyperparameters
    model = DecisionTreeRegressor(
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


def train_model_with_optuna(X_train, y_train, n_trials=20, plot_path="model_output/decision_tree_plot.png"):
    """Trains a Decision Tree model using Optuna for hyperparameter tuning and saves a plot."""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    print(f"Best trial parameters: {study.best_params}")
    print(f"Best trial score: {study.best_value}")
    
    # Train the final model with the best parameters
    best_params = study.best_params
    best_model = DecisionTreeRegressor(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    
    # Save the decision tree plot as an artifact
    save_decision_tree_feature_importance(best_model, X_train.columns, plot_path)
    
    return best_model, study


def train_decision_tree(X_train, y_train, max_depth=7):
    """Trains a Decision Tree Regressor with a specified max depth."""
    print("Training Decision Tree Regressor...")
    dtc_model = DecisionTreeRegressor(random_state=23, max_depth=max_depth)
    dtc_model.fit(X_train, y_train)
    print("Decision Tree training completed.")
    return dtc_model

def train_random_forest(X_train, y_train, 
                        n_estimators=500, 
                        min_samples_leaf=40, 
                        min_samples_split=90, 
                        max_features="sqrt"):
    """Trains a Random Forest Regressor with default or specified hyperparameters."""
    print("Training Random Forest Regressor...")
    RF_model = RandomForestRegressor(
        random_state=34,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        max_features=max_features
    )
    RF_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return RF_model

def tune_decision_tree(X_train, y_train):
    """Tunes hyperparameters for a Decision Tree Regressor using GridSearchCV."""
    print("Tuning Decision Tree Regressor with GridSearchCV...")
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    dtc = DecisionTreeRegressor(random_state=23)
    grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")
    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train):
    """Tunes hyperparameters for a Random Forest Regressor using RandomizedSearchCV."""
    print("Tuning Random Forest Regressor with RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    RF_model = RandomForestRegressor(random_state=34)
    random_search = RandomizedSearchCV(RF_model, param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_}")
    return random_search.best_estimator_