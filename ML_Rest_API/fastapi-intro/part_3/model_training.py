import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from model_utils import save_pickle

skf = StratifiedKFold(n_splits=5,random_state=67, shuffle=True)
    
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['log2', 'sqrt'])

    # Create a Random Forest model with the trial hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    # Perform cross-validation and return the mean auc
    score = cross_val_score(model, X_train_final, y_train, cv=skf, scoring='roc_auc').mean()

    return score

def run_optimization(X_train, y_train, n_trials=30):
    study = optuna.create_study(direction='maximize')
    objective = define_objective(X_train, y_train)
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)

    # Train final model using best hyperparameters
    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Save best model
    save_pickle(best_model, 'pickles/best_model.pkl')

    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))