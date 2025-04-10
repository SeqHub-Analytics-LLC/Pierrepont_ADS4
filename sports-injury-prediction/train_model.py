import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def train_model(X_train, y_train, params=None):
    def objective(trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()

        # log every trial
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            })
            mlflow.log_metric("accuracy", score)
            mlflow.set_tag("run_name", f"run-{child_run.info.run_id}")
        
        return score

    study = optuna.create_study(direction="maximize")
    with mlflow.start_run() as parent_run:  # start parent run
        study.optimize(objective, n_trials=50)

        # get best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")

        # rerain  final model with  best hyperparameters
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)

        # Log best model 
        mlflow.sklearn.log_model(model, "best_random_forest_model")
        mlflow.log_params(best_params)
        mlflow.set_tag("parent_run", "best_model")
    
    mlflow.end_run()
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")
    print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
    
    # Plot ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_prob)))
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    return predictions, cm
