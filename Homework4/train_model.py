import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_and_log_model(X_train, y_train, X_test, y_test):
    # Load the data

    # Define the objective function for Optuna
    def objective(trial):
        with mlflow.start_run(nested=True):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_params({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            })
            mlflow.log_metric("accuracy", accuracy)

            return accuracy

    # Start a parent MLflow run
    mlflow.set_experiment("breast_cancer_prediction")
    with mlflow.start_run(run_name="Parent Run") as parent_run:
        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, catch=(Exception,))

        

        # Log the best parameters and accuracy to MLflow
        best_params = study.best_params
        best_accuracy = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy)

        # Train the final model with the best parameters
        with mlflow.start_run(run_name="Final Model Training", nested=True):
            final_model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42
            )
            final_model.fit(X_train, y_train)

            # Save the final model
            with open('pickels/final_model.pkl', 'wb') as f:
                pickle.dump(final_model, f)

            mlflow.sklearn.log_model(final_model, "model")