import pandas as pd # type: ignore
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from utils import engineer_features, encode_features, scale_features


def model_creation(df):
  df = engineer_features(df)
  target_col = 'Likelihood_of_Injury'
  categorical_cols = ["BMI_Classification", "Age_Group","Training_Surface", "Position"]
  numeric_cols = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries',
              'Training_Intensity', 'Recovery_Time']

  X = df.drop(columns=[target_col])
  y = df[target_col]


  X = encode_features(X, categorical_cols)
  X = scale_features(X, numeric_cols)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  def objective(trial):
      n_estimators = trial.suggest_int('n_estimators', 50, 500)
      max_depth = trial.suggest_int('max_depth', 2, 10)
      min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
      max_features = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
      model = RandomForestClassifier(
          n_estimators=n_estimators,
          max_depth=max_depth,
          min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf,
          max_features=max_features,
          random_state=42
      )
      score = cross_val_score(model, X_train, y_train, cv=model, scoring='roc_auc').mean()

  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=30)

  best_params = study.best_params
  print("Best Hyperparameters:", best_params)

  model = RandomForestClassifier(**best_params, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__=="__main__":
    df=pd.read_csv("injury_data_with_categories.csv")
    model_creation(df)