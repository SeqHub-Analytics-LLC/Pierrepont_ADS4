from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import optuna
from sklearn.metrics import mean_squared_error

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=23, max_depth=7)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=34, n_estimators=500, min_samples_leaf=40, min_samples_split=90, n_jobs=-1, max_features="sqrt")
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(max_depth=3, random_state=85)
    model.fit(X_train, y_train)
    return model

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }

    model = GradientBoostingRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

def train_xgboost_with_optuna(X_train, y_train, X_val, y_val):

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    final_model = GradientBoostingRegressor(random_state=42, **best_params)
    final_model.fit(X_train, y_train)
    return final_model