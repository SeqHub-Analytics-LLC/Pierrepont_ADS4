import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from preprocessing import X_train, y_train, X_train_transformed
import numpy as np

# define a stratified split
skf = StratifiedKFold(n_splits=5,random_state=67, shuffle=True)

# Objective function to optimize
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

    # Perform cross-validation and return the mean accuracy
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy').mean()

    return score



# Objective function for Optuna
def objective(trial):
    # Hyperparameters for the neural network
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 32, 128, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

    fold_accuracies = []

    # Stratified K-Fold Cross-validation
    for train_index, val_index in skf.split(X_train_transformed, y_train):
        # Create the train and validation sets for this fold
        X_train_fold, X_val_fold = X_train_transformed[train_index], X_train_transformed[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Build the neural network model
        model = Sequential()

        # Input (Initial Layer)

        model.add(Input(shape=(X_train_transformed.shape[1],))) # Add input layer configuration
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(dropout_rate))

        # Add subsequent (hidden) layers
        for _ in range(n_layers - 1):
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(dropout_rate))

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=batch_size, verbose=0)

        # Evaluate the model on the validation set
        preds = model.predict(X_val_fold)
        pred_labels = (preds > 0.5).astype(int).flatten()  # Since predictions are probabilities, threshold at 0.5
        true_labels = y_val_fold.values.flatten()  # Ensure y_test is in the same format
        accuracy = accuracy_score(true_labels, pred_labels) # Calculate accuracy

        fold_accuracies.append(accuracy)

    # Return the mean accuracy across all folds
    return np.mean(fold_accuracies)
