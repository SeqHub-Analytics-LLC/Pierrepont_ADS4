import matplotlib.pyplot as plt
import optuna
import tkinter as tk

def print_metrics(rmse, r2):
    print("Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

def plot_param_importance(study):
    try:
        importance = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 5))
        plt.bar(importance.keys(), importance.values(), color="skyblue")
        plt.xlabel("Hyperparameters")
        plt.ylabel("Importance")
        plt.title("Hyperparameter Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig("model_output/param_importance.png")
    except Exception as e:
        print(f"Error plotting parameter importance: {e}")