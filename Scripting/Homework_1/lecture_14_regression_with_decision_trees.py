## Dataset
The BenchPress Weight dataset is a comprehensive collection of powerlifting metrics designed for the exploration and prediction of bench press performance. The dataset includes key variables that are typically associated with the bench press, which is one of the three lifts in the sport of powerlifting.

The main objective of the dataset is to facilitate the understanding and prediction of how much weight an athlete can bench press based on their body weight and performance in other lifts, specifically the squat and deadlift.

Key features of the dataset include:

 - `playerId`: A unique identifier for each athlete.
- `Name`: The name of the athlete.
- `Sex`: The sex of the athlete (male/female).
- `Equipment`: The type of equipment used by the athlete (e.g., raw, wraps, single-ply, multi-ply).
- `Age`: The age of the athlete.
- `BodyweightKg`: The body weight of the athlete in kilograms.
- `BestSquatKg`: The best squat weight lifted by the athlete in kilograms.
- `BestDeadliftKg`: The best deadlift weight lifted by the athlete in kilograms.
- `BestBenchKg`: The target variable for the regression analysis, which is the best bench press weight that an athlete can possibly handle.

Let's start by loading our dataset
"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir("drive/MyDrive/Pierrepont 2024/Intro to ML/datasets")

import pandas as pd
import numpy as np

#Load dataset
X_train = pd.read_csv("powerlifting_dataset/X_train.csv")
X_test = pd.read_csv("powerlifting_dataset/X_test.csv")
y_train = pd.read_csv("powerlifting_dataset/y_train.csv")
y_test = pd.read_csv("powerlifting_dataset/y_test.csv")

#Preview the dataset
X_train.head(4)

y_train.head(4)

X_test.head(4)

y_test.head(4)

"""In `y_test`, we have identified redundant instances of `Age`, `BodyweightKg`, and `BestDeadliftKg`. To ensure the integrity of our analysis and avoid the potential biases associated with duplicated data, we will proceed by removing these repeated entries."""

y_test.drop(columns = ["Age",	"BodyweightKg",	"BestDeadliftKg"],inplace=True)
y_test.head(4)

"""## Preprocess & Check Data

Next, it's important to note that both the feature dataframe (X) and the target dataframe (Y) share a common linking attribute, namely `playerId`. To ensure data integrity and consistency, we must verify that the `playerId` order is maintained across both the feature and target dataframes for the training and test sets. Preserving this sequence is crucial for maintaining the correct alignment between the data and their corresponding ground truth values.
"""

print(np.sum(X_test["playerId"].values == y_test['playerId']) == X_test.shape[0])
print(np.sum(X_train["playerId"].values == y_train['playerId']) == X_train.shape[0])

"""Let's take a deeper look at our dataset."""

X_train.info()

"""We noticed that the `BestSquatKg` feature is represented with the inapprpriate datatype. So let's investigate and then convert it to the appropriate data type."""

def convert(x):
    try:
        x = float(x)
    except:
        return x
    return x

BestSquatKg_dtype = [*map(lambda x : type(convert(x)) != float, X_train['BestSquatKg'].unique())]
BestSquatKg_dtype2 = [*map(lambda x : type(convert(x)) != float, X_test['BestSquatKg'].unique())]

wrong_train_data = X_train['BestSquatKg'].unique()[BestSquatKg_dtype]
wrong_test_data = X_test['BestSquatKg'].unique()[BestSquatKg_dtype2]

wrong_train_data, wrong_test_data

"""It seems that the test data doesn't contain the erroneous entries. We can observe that the erroneous entries have additional dots in some number. We can simply just remove this additional dots to move on."""

correct_train_data = np.array([*map(lambda x : x[:-2] + x[-1], wrong_train_data)])
correct_train_data = correct_train_data.astype('float')
for i in range(len(wrong_train_data)):
    X_train.loc[X_train['BestSquatKg'] == wrong_train_data[i], 'BestSquatKg'] = correct_train_data[i]

X_train['BestSquatKg'] = X_train['BestSquatKg'].astype('float')
X_train['BestSquatKg'].dtype

"""Additionally, we can see that the `Age` feature contains some missing values."""

X_train['Age'].describe()

# Plot histogram
import seaborn as sns
sns.histplot(data = X_train['Age'], kde = True)

"""Based on the histogram, the data for `Age` appears to be right-skewed, which means that there are more observations at the lower end of the age spectrum and fewer as age increases. We can see that the peak of the distribution is towards the younger end, with a tail stretching out towards the older ages.

A good choice for replacing the missing values in the `Age` would be mean, as it is more robust to outliers and may represent the central tendency of the distribution more effectively.

- **Note:** To prevent data leakage, **it is crucial that we use the mean age derived from the training set to impute missing age values in the test set**. This ensures that the imputation strategy does not inadvertently incorporate information from the test set, which should remain unseen during the model training process.
"""

mean_age = X_train['Age'].mean()

#handle train data
X_train['Age'].fillna(mean_age,inplace=True)
#handle test data
X_test['Age'].fillna(mean_age,inplace=True)

"""#### Check for Duplicates
Next, we will check the dataset to see if there are any duplicated rows.
"""

print(X_train.duplicated().sum())
print(X_test.duplicated().sum())

"""There are no duplicates in our dataset.

Before going into Exploratory Data Analysis. We should join the target and input features in our training and testing set.
"""

input_features = pd.concat([X_train,X_test], axis=0)
targets = pd.concat([y_train,y_test], axis=0)

input_features.describe()

"""We can see that some of the weight related features are negative , which is not possible. It is possible that these values were entered wrongly with a negative sign. So, we will convert them to positive by taking the absolute value."""

kg_features = input_features.filter(regex='Kg').columns
input_features[kg_features] = np.abs(input_features[kg_features])

"""The `name` and `playerId` features has no significant value in our analysis. Hence, we will drop these columns."""

input_features.drop(columns =["Name","playerId"],inplace=True)
targets.drop(columns =["playerId"],inplace=True)

"""## Exploratory Data Analysis

In this section, you will perform Exploratory Data Analysis (EDA) on the provided powerlifting dataset. We have outlined five questions for you to investigate. For each question, we've provided hints to help you approach your analysis effectively. Utilize plots and statistical summaries where appropriate to draw insights from the data.
"""

train_ = input_features.iloc[:X_train.shape[0],:]
target_ = targets.iloc[:X_train.shape[0]]

"""### Question 1: What is the distribution of BestBenchKg among the lifters?

*Hint: Consider plotting a histogram or density plot to visualize the distribution of `BestBenchKg`. Pay attention to the shape, spread, and any potential outliers in the data.*

### Solution
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot the distribution of BestBenchKg
plt.figure(figsize=(10, 6))
sns.histplot(target_['BestBenchKg'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Best Bench Press (Kg)')
plt.xlabel('Best Bench Press (Kg)')
plt.ylabel('Frequency')
plt.show()

"""#### Insights
- The distribution appears to be bi-modal, indicating that there might be two groups of athletes: those who can benchpress between 0 and 100kg and those who can benchpress over 100kg.
- The long tail to the right suggests that there are some lifters who achieve exceptionally high bench press weights, but they are relatively rare.

### Question 2: How does the BestBenchKg vary between male and female lifters?

*Hint: Use box plots to compare the distribution of `BestBenchKg` between male and female lifters. Look for differences in the central tendency, variability, and range.*
"""

## Your implementation here

"""### Solution

"""

# Plot the distribution of BestBenchKg between male and female lifters
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_['Sex'], y=target_['BestBenchKg'], palette='coolwarm')
plt.title('Comparison of Best Bench Press (Kg) Between Male and Female Lifters')
plt.xlabel('Sex')
plt.ylabel('Best Bench Press (Kg)')
plt.show()

"""From the plot, we can observe the following:

1. The median `BestBenchKg` for male lifters is higher than that for female lifters, as indicated by the position of the horizontal line within each box.
2. The interquartile range (IQR), represented by the height of each box, is wider for male lifters, suggesting more variability in bench press performances among men compared to women.
3. Both distributions have outliers, as indicated by the points outside the whiskers of each box plot, but the male lifters' outliers extend to much higher `BestBenchKg` values.
4. The overall range of `BestBenchKg`, from the minimum to the maximum excluding outliers, is broader for male lifters.

### Question 3: Is there a relationship between BodyweightKg and BestBenchKg?

*Hint: Create a scatter plot with a trend line to explore the relationship between BodyweightKg and BestBenchKg. Additionally look at the correlation coefficient to determine the strength of the relationship. Consider how body weight might influence bench press performance.*
"""

#Your implementation here

"""### Solution"""

#solution
plt.figure(figsize=(12, 8))
sns.regplot(x=train_['BodyweightKg'], y=target_['BestBenchKg'], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between Bodyweight and Best Bench Press')
plt.xlabel('Bodyweight (Kg)')
plt.ylabel('Best Bench Press (Kg)')
plt.show()


print(f"Correlation Matrix between Bodyweight and Best Bench Press\n {np.corrcoef(train_['BodyweightKg'],target_['BestBenchKg'])}")

"""From the plot, we can observe the following:

1. There appears to be a positive correlation between `BodyweightKg` and `BestBenchKg`, as indicated by the upward slope of the trend line. This suggests that, on average, lifters with higher body weights tend to have better bench press performances.
2. The scatter of points shows a wide range of `BestBenchKg` values at most body weights, indicating variability in bench press performance that cannot be explained by body weight alone.
3. The density of points is higher at lower body weights, suggesting that there are more lifters within these weight categories in the dataset.

### Question 4: How does the choice of Equipment (e.g., Raw, Wraps) affect `BestBenchKg`?

*Hint: Compare the average or median BestBenchKg across different equipment categories using bar plots. Reflect on how equipment might enhance or affect performance.*
"""

## Your Implementation Here

"""### Solution"""



#solution
plt.figure(figsize=(10, 6))
sns.barplot(x=train_['Equipment'], y=target_['BestBenchKg'], ci=None, palette='Set2')
#sns.boxplot(x=train_['Equipment'], y=target_['BestBenchKg'], palette='Set2')
plt.title('Impact of Equipment on Best Bench Press (Kg)')
plt.xlabel('Equipment')
plt.ylabel('Average Best Bench Press (Kg)')
plt.show()

"""From the plot, we can observe:

1. There are noticeable differences in the average `BestBenchKg` across the different equipment categories. Some categories show higher average bench press weights than others, suggesting that the choice of equipment could influence bench press performance.
2. The equipment category labeled `Raw` likely represents lifters who do not use supportive gear like bench shirts or wraps, while other categories might indicate the use of various supportive equipment, which can aid in lifting heavier weights.

### Question 5: Is there a noticeable difference in BestBenchKg across different age groups?

*Hint: Creating age groups: { (18 and under), (19-23), (24-38), (39-49), (50-59), (60-69), and (70+)} and analyze the average `BestBenchKg` within each. Use bar plots or box plots to visualize how age may impact performance*
"""

## Your implementation here

"""### Solution"""

# Define age bins and labels
age_bins = [0, 18, 23, 38, 49, 59, 69, float('inf')]
age_labels = ['18 and under', '19-23', '24-38', '39-49', '50-59', '60-69', '70+']

# Categorize the data into age groups
train_['AgeGroup'] = pd.cut(train_['Age'], bins=age_bins, labels=age_labels, right=False)

# Plot the average BestBenchKg across age groups
plt.figure(figsize=(12, 8))
sns.barplot(x=train_['AgeGroup'], y= target_['BestBenchKg'], ci=None, palette='viridis')
plt.title('Best Bench Press (Kg) Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Average Best Bench Press (Kg)')
plt.xticks(rotation=45)
plt.show()

"""From the plot, we can observe:

1. There are differences in the average BestBenchKg across age groups, indicating that age might have an impact on bench press performance.
2. The age groups of "24-38" and "39-49" seem to have higher average BestBenchKg compared to the younger and older age groups. This could suggest that lifters in these age ranges are, on average, stronger or more experienced in bench pressing.
3. The youngest (18 and under) and the oldest (70+) age groups tend to have lower average BestBenchKg, which could reflect less muscle mass or strength due to age, less training experience in the case of younger lifters, or age-related decline in the case of older lifters.

## Feature Engineering

Feature engineering is a critical step in improving model performance, especially in regression tasks where the relationships between features can provide meaningful insights. Although our initial dataset includes `Sex`, `Equipment`, `Age`, `BodyweightKg`, `BestSquatKg`, `BestDeadliftKg`, and the target `BestBenchKg`. It is possible to engineer new features.
"""

data_df = pd.concat([input_features,targets],axis=1)
data_df.head()

"""1. **Relative Strength:** Calculate the relative strength index by dividing the lift weights (BestSquatKg and BestDeadliftKg) by the BodyweightKg. This normalizes the lifts by body weight, which is often a more accurate indicator of strength.



"""

# Calculate relative strength by dividing the lift weights by the body weight
data_df['RelativeSquatStrength'] = data_df['BestSquatKg'] / data_df['BodyweightKg']
data_df['RelativeDeadliftStrength'] = data_df['BestDeadliftKg'] / data_df['BodyweightKg']

"""2. **Equipment Index:** In powerlifting, the type of equipment used can significantly influence a lifter's performance due to the support and mechanical advantage provided by the gear. We can rank the different equipment types based on known impact on bench press performance. We will assign an index or score to each equipment type based on their expected influence on lifting ability. 'Multi-ply' equipment is considered to provide the most support, followed by 'Single-ply', then 'Wraps', with 'Raw' providing no additional support.

"""

# Define the mapping from equipment type to index
equipment_scores = {
    'Raw': 1,
    'Wraps': 2,
    'Single-ply': 3,
    'Multi-ply': 4
}

# Map the equipment types to their respective scores
data_df['Equipment_Index'] = data_df['Equipment'].map(equipment_scores)

"""3. **Categorical Variable Encoding**: We will endoe the `Sex` and `Equipment` variables using One hot Encoding.

"""

#Check unique categories in categorical columns
data_df["Equipment"].unique(), data_df['Sex'].unique()

"""Next we will use `pd.get_dummies` to convert categorical variable into dummy/indicator variables. We will also drop the first category to avoid the dummy variable trap."""

data_df = pd.get_dummies(data_df, columns=['Equipment',"Sex"], drop_first=True)
data_df.head()

"""4. **Age Category**: Instead of using the continuous Age feature, we could categorize athletes into age groups. The International Powerlifting Federation (IPF) uses the following age categories:
- sub-junior (18 and under)
- junior (18-23)
- open (24-38)
- masters 1 (39-49)
- masters 2 (49-59)
- masters 3 (59-69)
- masters 4 (69+)

This can help capture different performance trends across age ranges.
"""

# Define the bins for the age categories
bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

# Bin the age data
data_df['AgeCategory'] = pd.cut(data_df['Age'], bins=bins, labels=labels, right=False)
data_df['AgeCategory']

"""Now we will encode these categories using scikit-learn's [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

Ordinal encoding is a feature transformation technique used to convert categorical data that has a natural order or ranking (like 'low', 'medium', 'high') into numerical format. This process assigns each unique category value to an integer based on their order or rank, which is essential for allowing algorithms to interpret the data during modeling.
"""

# Now we will use OrdinalEncoder to encode these categories
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[labels])
data_df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(data_df[['AgeCategory']])
data_df['AgeCategoryEncoded']

data_df.head(3)

"""## Model Development: Building the Regression Model

For the sake of inference speed and hardware limitation. We will restrict the test set to the last 5000 examples.
"""

target = data_df['BestBenchKg']
input_features = data_df.drop(columns =["BestBenchKg","Age","AgeCategory"])

#Targets
y_train, y_test = target.iloc[:X_train.shape[0]],target.iloc[-5000:]
#Input features
X_train, X_test =  input_features.iloc[:X_train.shape[0],:], input_features.iloc[-5000:,:]

print(X_train.shape, X_test.shape), print(y_train.shape, y_test.shape)

"""In order to measure the performance of our models, we will utilize evaluation metrics such as the root mean squared error and the R2 score."""

from sklearn.metrics import mean_squared_error, r2_score

def evaluate(train_predictions, test_predictions):

    r2_train = r2_score(train_predictions, y_train)
    r2_test = r2_score(test_predictions, y_test)
    print(f"R2:\nTrain: {r2_train}, Test: {r2_test}")

    rmse_train = np.sqrt(mean_squared_error(train_predictions, y_train))
    rmse_test = np.sqrt(mean_squared_error(test_predictions, y_test))
    print(f"RMSE:\nTrain: {rmse_train}, Test: {rmse_test}")

"""We will also create a function `plot_feature_importance`, this function will help us visualize the most relevant features towards prediction."""

from matplotlib import pyplot as plt
def plot_feature_importance(importance, names, model_name):
    #create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # create a dataframe using a dictionary
    data = {'feature_names':feature_names, 'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'],ascending = False, inplace=True)

    #define size of bar plot
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y = fi_df['feature_names'])
    plt.title(model_name + "FEATURE IMPORTANCE")

    #Add chart labels
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")

"""### **Class Discussion**

---

> **Do tree-based regression models like Decision Trees and Random Forests require feature scaling to perform effectively?**

Feature scaling, which involves standardizing or normalizing features, is not strictly necessary for tree-based regression models such as Decision Trees and Random Forests. This is because these models make decisions based on thresholds and do not compute distances between points or assume a particular distribution of the features. As a result, the scale of the features does not impact the model's ability to split nodes and make predictions. However, in some scenarios, especially when integrating tree-based models with other algorithms that do require feature scaling, or when using certain visualization or analysis techniques, scaling might still be beneficial for consistency or interpretability.

## Tree-Based Regression Models

Tree-based regression models are a subset of machine learning algorithms that use a decision tree structure to predict continuous outcomes based on input features. These models split the data into subsets using decision rules inferred from the features, effectively breaking down a complex regression problem into simpler, more manageable parts. The final prediction for a given input is typically the average target value of the training instances within the same leaf node.

Key characteristics include:

1. **Hierarchical Structure:** The decision tree starts with a root node and branches out into internal nodes and leaf nodes, representing decision points and outcomes, respectively.

2. **Non-Linearity:** Tree-based models can capture non-linear relationships between features and the target variable without requiring transformations.

3. **Feature Importance:** These models can inherently provide insights into the importance of each feature in predicting the target variable.

4. **Versatility:** They can handle both numerical and categorical data and do not require feature scaling.


Common tree-based regression models include:

1. **Decision Tree Regression:** Utilizes a single decision tree, making it simple and interpretable but prone to overfitting.

2. **Random Forest Regression:** Builds multiple decision trees and aggregates their predictions to improve accuracy and robustness.

3. **Gradient Boosting Trees:** Constructs sequential trees where each tree attempts to correct the errors of the previous one, often leading to high performance.


For a recap on the inner working of these algorithms, please revisit [lecture 05](https://colab.research.google.com/drive/1Yd34Jg3bnV3kD3n20XfBnGU3iMyIuyt_)

- **Note:** <font color="darkred">Tree-based regression models are widely used due to their interpretability, ease of use, and effectiveness in handling various types of data. However, they can be susceptible to overfitting, especially with complex trees, and might require careful tuning of hyperparameters to achieve the best performance.</font>

### Decision Trees
"""

from sklearn.tree import  DecisionTreeRegressor

dtc_model =  DecisionTreeRegressor(random_state=23, max_depth=7)

dtc_model.fit(X_train, y_train)

dtc_train_predictions = dtc_model.predict(X_train)
dtc_test_predictions = dtc_model.predict(X_test)

evaluate(dtc_train_predictions, dtc_test_predictions)

importances = dtc_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

"""Let's visualize the top features.

"""

plot_feature_importance(importances , names = X_train.columns, model_name= "Decision Trees")

"""Given our model performance, there is still room for improvement.
#### Hyper-parameter Optimization

- **Task:** Your task is to enhance the performance of your Decision Tree model by utilizing either RandomizedSearchCV or GridSearchCV for hyperparameter optimization. Begin by selecting the parameters you wish to tune. After determining the optimal parameter values through the search process, instantiate a new Decision Tree model using these optimized parameters. Then, evaluate and compare the performance of this optimized model against the original to assess the improvement achieved.

Find the list of parameters in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).
"""

#To-Do: Hyper-parameter Optimization

"""### Random Forests"""

from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(random_state=34,n_estimators=500, min_samples_leaf=40,
                                 min_samples_split=90, n_jobs=-1, max_features="sqrt")

RF_model.fit(X_train, y_train)

RF_train_predictions = RF_model.predict(X_train)
RF_test_predictions = RF_model.predict(X_test)

evaluate(RF_train_predictions, RF_test_predictions)

importances = RF_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

"""Let's visualize the top features.

"""

plot_feature_importance(importances , names = X_train.columns, model_name= "Random Forests ")

"""Given our model performance, there is still room for improvement.
#### Hyper-parameter Optimization

- **Task:** Your task is to enhance the performance of your Random Forest model by utilizing either RandomizedSearchCV or GridSearchCV for hyperparameter optimization. Begin by selecting the parameters you wish to tune. After determining the optimal parameter values through the search process, instantiate a new Random Forest model using these optimized parameters. Then, evaluate and compare the performance of this optimized model against the original to assess the improvement achieved.

Find the list of parameters in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html_).
"""

#To-Do: Hyper-parameter Optimization

"""### Gradient Boosting Trees

Gradient Boosting is a machine learning technique that builds models incrementally in a stage-wise fashion. It combines the predictions from multiple weak learners (typically decision trees) to create a strong learner that performs better than any of the individual models. The concept of boosting involves sequentially adding weak learners, where each new model focuses on correcting the errors made by the previous ones. In Gradient Boosting, this correction is guided by the gradient of the loss function, which indicates the direction in which the model should be improved, hence the name "Gradient" Boosting.


The implementation in scikit-learn has a lot of overlapping parameters with Random Forests. The hyper-parameters have the same meaning and functions.
"""

from IPython.display import YouTubeVideo

YouTubeVideo('zTXVo2Kmi8Q', width=800, height=500)

from sklearn.ensemble import GradientBoostingRegressor

GBM_model = GradientBoostingRegressor(max_depth=3,random_state=85)

GBM_model.fit(X_train, y_train)

GBM_train_predictions = GBM_model.predict(X_train)
GBM_test_predictions = GBM_model.predict(X_test)

evaluate(GBM_train_predictions, GBM_test_predictions)

importances = GBM_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

"""Let's visualize the top features.

"""

plot_feature_importance(importances , names = X_train.columns, model_name= "Gradient Boosting Trees")

"""Given our model performance, there is still room for improvement.
#### Hyper-parameter Optimization

- **Task:** Your task is to enhance the performance of your Gradient Boosting model by utilizing either RandomizedSearchCV or GridSearchCV for hyperparameter optimization. Begin by selecting the parameters you wish to tune. After determining the optimal parameter values through the search process, instantiate a new Gradient Boosting model using these optimized parameters. Then, evaluate and compare the performance of this optimized model against the original to assess the improvement achieved.

Find the list of parameters in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
"""

#To-Do: Hyper-parameter Optimization

"""### Conclusion


In conclusion, this lecture has provided a comprehensive guide on employing tree-based regression models, specifically focusing on Decision Trees, to predict the bench press weights of powerlifting athletes. We've journeyed through crucial steps, starting from data preprocessing, where we prepared the BenchPress Weight dataset for analysis, to feature selection, where we identified the most relevant variables influencing bench press performance. We then progressed to model training, where we constructed our regression models, followed by hyperparameter tuning to optimize its performance. Finally, we evaluated our model to assess its predictive accuracy.

As we wrap up this lesson, we set the stage for your upcoming term project, where you'll apply the principles of regression analysis to a new dataset. This project will challenge you to integrate the knowledge and skills you've acquired, pushing you to explore innovative approaches to regression problems.

Additionally, lesson review notes on regression will soon be released. These notes will serve as a valuable resource, summarizing key concepts and techniques discussed throughout our lectures on regression, and providing you with a reference point to solidify your understanding and aid in your project work.
"""