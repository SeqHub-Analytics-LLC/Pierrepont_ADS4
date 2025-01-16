# **Assignment: Reproducing a Modular Script System from a Colab Notebook**

## **Overview**

In this assignment, you will transform the provided Colab notebook into a modular Python script-based system. The goal is to organize the code into multiple scripts, following a well-structured directory hierarchy. This approach promotes reusability, modularity, and scalability of machine learning projects.

## **Objectives**

1. Convert the provided Colab notebook into a modular Python script system.
2. Create and organize scripts into appropriate categories, including data processing, model training, validation, and utilities.
3. Implement a pipeline to process the data, train the model, and evaluate its performance.
4. Use Optuna for hyperparameter optimization instead of traditional grid search.

---

## **Directory Structure**

Your project should follow the directory structure below:

```
Homework/
├── data_processing.py         # Functions for data cleaning, transformation, and feature engineering
├── model_training.py          # Script for model training using Optuna for hyperparameter optimization
├── model_validation.py        # Functions to validate and evaluate trained models
├── main.py                    # Entry point to run the entire pipeline
├── utils.py                   # Helper functions (e.g., metrics, Optuna study visualization)
├── evaluate.py                # Script for testing and evaluating saved models
├── preprocessed_data/         # Directory for storing preprocessed datasets
├── model_output/              # Directory for storing outputs related to modeling
├── test_data.csv              # Example eval dataset (optional, for evaluation)
├── Power_lifting.csv           # Original dataset (provided)
```

---

## **Assignment Steps**

### **Step 1: Setup**
1. Download the provided (Colab notebook)[https://colab.research.google.com/drive/1cylgzMWrVvYjSiLzGzwLBHLuw-zvc3d7?usp=sharing] and dataset (Power lifting Dataset)[].
2. Place all scripts in the Homework folder and structure it as shown above.
3. You don't need to upload a requirement file anymore.
---

### **Step 2: Create Scripts**

#### 1. **`data_processing.py`**
   - Include functions to clean, preprocess, and scale the data.
   - Extract relevant features (e.g., date features).
   - Save preprocessed datasets into the `preprocessed_data/` directory.

#### 2. **`model_training.py`**
   - Implement the Decision Tree training pipeline.
   - Use Optuna for hyperparameter tuning.
   - Save the trained model and Optuna study into the `model_output/` directory.

#### 3. **`model_validation.py`**
   - Write functions to evaluate the trained model on the test dataset.
   - Include metrics like RMSE and R².

#### 4. **`utils.py`**
   - Add helper functions, such as:
     - Printing metrics.
     - Visualizing Optuna study results (e.g., parameter importance).

#### 5. **`evaluate.py`**
   - Load the saved model and test it on new datasets (e.g., `test_data.csv`).
   - Print the evaluation metrics.

#### 6. **`main.py`**
   - Serve as the entry point for the entire pipeline.
   - Load the dataset, preprocess it, train the model, and evaluate the results.

---

### **Step 3: Reproduce Results**
1. Your system should preprocess the provided `Apple_demand.csv` dataset, split it into training and testing sets, and train a Random Forest model.
2. Optimize the model using Optuna.
3. Save the model and evaluation results.
4. Verify your system by running `evaluate.py` on `test_data.csv`.

---

## **Deliverables**

1. **Codebase**:
   - All scripts (`data_processing.py`, `model_training.py`, etc.).
   - Organized directory structure.

2. **Documentation**:
   - Comments within the code explaining each step.
   - A summary in a `explanation.md` describing:
     - How your code is structured.
     - Instructions to run the pipeline.
     - Any challenges faced during implementation.

---

## **Running the Project**

To execute your pipeline:
1. Ensure all dependencies are installed (`pandas`, `sklearn`, `optuna`, `joblib`).
2. Run `main.py`:
   ```bash
   python main.py
   ```
3. Test the saved model using `evaluate.py`:
   ```bash
   python evaluate.py
   ```

---

## **Submission**

- In your own fork of the class repository. Place your solution within the Homework.
- Include:
  - All Python scripts.
  - The `README.md`.
  - Any additional files required to run your pipeline.

