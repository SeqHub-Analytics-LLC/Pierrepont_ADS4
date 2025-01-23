import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """Saves preprocessed datasets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


def save_model(model, output_path):
    """Saves the trained model."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    joblib.dump(model, output_path)


def save_encoder(model, output_path):
    """Saves the encoder."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    joblib.dump(model, output_path)


def save_decision_tree_feature_importance(model, feature_names, plot_path):
    """
    Saves a plot of the feature importances from the trained Decision Tree model.
    """
    # Extract feature importances and sort them
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order
    
    # Arrange features and their importance scores
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features, sorted_importances, color='skyblue', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Feature Importances in Decision Tree', fontsize=16)
    plt.gca().invert_yaxis()  # To display the most important feature on top
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")