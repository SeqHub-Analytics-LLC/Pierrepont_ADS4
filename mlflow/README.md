# From Notebooks to Scripts: Writing Modular Python Code

# Welcome to this new module! 🚀

So far, we’ve been building models and tracking experiments using Jupyter Notebooks. While notebooks are great for prototyping and exploration, production-ready machine learning models require a more structured approach. This module will teach you how to convert your notebooks into modular Python scripts, preparing you to develop models suitable for deployment.


# Why Move from Notebooks to Scripts?

- **Modularity:** Scripts allow for cleaner, reusable code by organizing functionality into functions and modules.
- **Scalability:** Large projects can be broken into smaller, manageable components (e.g., data processing, model training, and evaluation).
- **Deployment Readiness:** Scripts are essential for integrating ML models into production systems (e.g., APIs, batch processing).
- **Version Control:** Scripts are easier to track, review, and manage in collaborative workflows using Git.


# Learning Objectives

By the end of this module, you will be able to:
  - **Refactor Jupyter Notebooks into Python Scripts:** Understand how to structure scripts for readability and maintainability.
  - **Write Modular Code:** Break down code into functions, classes, and modules.
  - **Include Essential Components:** Build scripts with clear sections for data preprocessing, model training, validation, and evaluation.
  - **Integrate MLflow in Scripts:** Replace notebook-based MLflow tracking with script-based logging.
  - Run and Test Scripts: Learn how to execute and debug Python scripts in the terminal or IDEs like VS Code.


# Module Outline

1. **Refactoring Basics**

- Identify repetitive code and refactor it into reusable functions.
- Convert cells in a Jupyter notebook into logical script sections:
  - **Imports:** Import all required libraries and modules at the top.
  - **Main Functionality:** Organize key operations (e.g., data loading, training, evaluation) into functions.

2. **Script Structure**

A good script structure might look like this:
```
  project/
  │
  ├── data_processing.py  # Functions for data cleaning, transformation, and feature engineering
  ├── model_training.py   # Script for model training and logging
  ├── model_validation.py # Functions to validate and evaluate models
  ├── main.py             # Entry point to run the entire pipeline
  │
  └── requirements.txt    # Dependency file
```

3. **Step-by-Step Conversion**

- **Export Code:** Export the notebook code to a .py file:
	- From the Jupyter Notebook interface:
    ```File > Download as > Python (.py)```
  - Or using the terminal:
    ```python
    jupyter nbconvert --to script notebook_name.ipynb
    ```

4. **Refactor into Modules:** Split the .py file into logical sections and move each section to its respective module (e.g., data_processing.py, model_training.py).

5.	**Add Entry Points:** Use construct below in your main script (main.py) to specify the entry point
  ```python 
  if __name__ == "__main__": 
  ```

	4.	Integrate MLflow Logging: Replace notebook cells with MLflow logging commands, ensuring everything is tracked as part of a cohesive pipeline.

4. **Running Scripts**

- Run scripts from the terminal:
  ```python
  python main.py
  ```
- Use arguments with scripts to make them configurable (e.g., using argparse).

## Best Practices for Modular Programming

- **Keep Code DRY (Don’t Repeat Yourself):**
  - Extract repetitive code into reusable functions or classes.
- **Use Meaningful Names:**
  - Name variables, functions, and files descriptively (e.g., train_model() instead of func1()).
- **Document Your Code:**
  - Add docstrings for functions and comments to explain complex logic.
- **Log Everything:**
  - Log key steps, metrics, and parameters using MLflow or Python’s logging module.
- **Version Control:**
	- Use Git to track changes, and make commits for each significant update.

## Deliverables for Students

- Refactor an existing Jupyter Notebook into a modular Python script.
	- Include the following components in your script:
	  - Data processing.
	  - Model training with MLflow logging.
	  - Validation and evaluation.
- Push the completed scripts to the central course repository.

## Support and Resources

To help you get started:
  - Python Packaging and Modules
  - VS Code Debugging Guide
  - MLflow Scripting Documentation

## Let’s Get Started!

Let’s dive into scripting and make our code production-ready! If you have any questions, feel free to reach out on Google Classroom.

Happy coding! 🚀
- **Taiwo Togun**
