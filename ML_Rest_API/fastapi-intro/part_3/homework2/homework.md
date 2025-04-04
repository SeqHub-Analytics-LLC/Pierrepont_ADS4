# 🧠 Homework 2: Injury Prediction Model (Python Scripting Practice)

This assignment challenges you to **convert a Colab notebook into modular Python scripts** — reinforcing your understanding of scripting, model building, and clean code organization. 

---

## 📌 Objective

The primary goal of this project is to build a **classification model** that predicts the **likelihood of an athlete getting injured**. The model will use features such as:

- Height and weight  
- Previous injuries  
- Recovery time  
- Training intensity  
- And other related variables

In the long run, this project will serve as the foundation for building a full machine learning pipeline — including **model deployment with FastAPI** in future exercises.

---

## 🛠️ Task Breakdown

Your task is to break down the Colab notebook into **organized Python files**. You should maintain clear structure, modularity, and separation of concerns.

---

### 🗂️ Recommended Project Structure

```
part_3/
└── homework2/
    ├── data/
    │   └── injury_data.csv        # Input dataset (if not downloading within script)
    ├── └── pickles                # Pickle files created.
    ├── preprocessing.py           # Data cleaning, feature engineering, encoding, scaling
    ├── train_model.py             # Model training and evaluation logic
    ├── model_utils.py             # Save/load model, encoder, scaler as .pkl
    ├── config.py                  # Constants like feature lists, file paths
    ├── main.py                    # Main script to run the full pipeline end-to-end
    └── homework.md                # Assignment instructions (this file)
   
    {Add files as you deem fit, do not be restricted to my suggestions}
```

---

## ✍️ Submission Instructions

- Push all relevant files to the directory:  
  ```
  part_3/homework2/
  ```
- Your code should be runnable by simply executing `main.py`

---

## 🚀 Future Use

This project is not just a one-off — you'll continue to build on it by:

- Serving the trained model as a **FastAPI service**
- Creating **API endpoints** for prediction and monitoring

---

## 📎 Reference

Colab source: [Homework 2 Notebook](https://colab.research.google.com/drive/1Dz0TW229ytWAHzxS8icuZkCbkckFdu0C?usp=sharing)

---

Good luck converting and scripting — this is your next step toward building real-world machine learning systems! 🚀
```
