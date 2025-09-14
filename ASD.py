#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Dependencies should be installed manually:
# pip install catboost missingno xgboost


# In[12]:


# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import missingno as msno

# Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

from scipy.stats import chi2_contingency
import scipy.stats as stats

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example DataFrame
data = {
    'age': [23, 45, 31, 35, 50, 29, 41],
    'result': [88, 92, 95, 70, 85, 90, 78]
}
df = pd.DataFrame(data)

# List of numerical columns
numeric_cols = ['age', 'result']

# Set the figure size
plt.figure(figsize=(12, 4))

# Calculate the number of rows and columns for the subplots
num_plots = len(numeric_cols)
num_cols = 3  # You can keep this as 3 or change as needed
num_rows = (num_plots // num_cols) + (1 if num_plots % num_cols != 0 else 0)

# Plot boxplots dynamically based on the number of features
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")

# Adjust layout
plt.tight_layout()
plt.show()


# In[14]:


df = pd.read_csv("C://Users//Madhura Sharan//OneDrive//Desktop//Major Project//ASDD//Autism Screening.csv");


# In[15]:


df.shape


# In[16]:


# display all columns of a dataframe
pd.set_option('display.max_columns', None)
df.head()


# In[17]:


df.info()


# In[18]:


# Replace '?' with NaN
df["age"] = df["age"].replace('?', np.nan)

# Convert to float (because NaN can't exist in int)
df["age"] = df["age"].astype(float)

# Fill NaNs with median and assign back properly (no inplace)
df["age"] = df["age"].fillna(df["age"].median())

# Convert to int finally
df["age"] = df["age"].astype(int)


# In[19]:


df.describe()


# In[20]:


# Count missing values in each column:
print("Missing Values:\n", df.isnull().sum())


# In[21]:


#Check for duplicate entries:
print("Duplicates:", df.duplicated().sum())


# In[22]:


df = df.drop_duplicates()


# In[23]:


print("Duplicates:", df.duplicated().sum())


# In[ ]:





# In[24]:


numerical_features = ["ID", "age", "result"]

# Header for the table
print("\n{:<20} {}".format("Column", "Unique Values"))
print("=" * 50)

# Loop through DataFrame columns
for col in df.columns:
    if col not in numerical_features:
        # Get all unique values
        unique_values = df[col].unique()
        unique_values_str = ", ".join(map(str, unique_values))

        # Wrap the text if it's too long
        print(f"{col:<20}", end="")
        print(f"{unique_values_str:<60}")
        print("-" * 50)


# In[25]:


# Display the unique values and their counts
unique_values_counts = df['ethnicity'].value_counts()
print(unique_values_counts)
print("\n")

# Display the unique values and their counts
unique_values_counts = df['relation'].value_counts()
print(unique_values_counts)


# In[26]:


df = df.drop(columns=["age_desc"])


# In[27]:


df.shape


# In[28]:


df.columns


# In[29]:


# Replace "?" with NaN using df.replace, and treat it as missing without modifying the original DataFrame
fig, ax = plt.subplots(figsize=(8, 4))
msno.matrix(df.replace("?", np.nan), ax=ax, sparkline=False)
plt.title("Missing Values Matrix (Treating '?' as Missing)", fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
plt.show()


# In[30]:


# Replace '?' in 'ethnicity' column with 'Unknown'  ( Since '?' represents about 25% of the total data )
df['ethnicity'] = df['ethnicity'].replace('?', 'Unknown')

# Display the updated 'ethnicity' column to check the changes
print(df['ethnicity'].value_counts())


# In[31]:


# Replace '?' in 'relation' column with the mode ('Self')  (Since '?' represents only 5% of the total data in 'relation')
most_frequent_relation = df['relation'].mode()[0]
df['relation'] = df['relation'].replace('?', most_frequent_relation)

# Display the updated 'relation' column to check the changes
print(df['relation'].value_counts())


# In[32]:


# define the mapping dictionary for country names
mapping = {
    "AmericanSamoa": "United States",
    "Hong Kong": "China"
}

# repalce value in the country column
df["contry_of_res"] = df["contry_of_res"].replace(mapping)


# In[33]:


df["contry_of_res"].unique()


# In[34]:


# taget class distribution
print(df["Class/ASD"].value_counts())


# In[35]:


# Plot the class distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class/ASD', data=df)
plt.title('Class Distribution')
plt.xlabel('Class/ASD')
plt.ylabel('Count')
plt.show()


# In[36]:


categorical_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'gender', 'jaundice', 'austim',
                    'used_app_before', 'relation']
num_cols=['age','result']


# In[37]:


# Descriptive statistics for numerical columns
df[num_cols].describe()


# In[38]:


features_to_plot = ["age", "result"]

# Create histograms for the selected features
df[features_to_plot].hist(figsize=(10, 5), bins=20)  # Adjust bins as needed
plt.suptitle("Distribution of Selected Features", fontsize=16)
plt.show()


# In[39]:


# List of numerical columns you want to plot
numeric_cols = ['age', 'result']

# Set the figure size
plt.figure(figsize=(12, 4))

# Calculate the number of rows and columns for the subplots
num_plots = len(numeric_cols)
num_cols = 3  # Since you have 3 numerical columns
num_rows = (num_plots // num_cols) + (1 if num_plots % num_cols != 0 else 0)

# Plot boxplots dynamically based on the number of features
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Plot count plots
plt.figure(figsize=(20, 12))
plotnumber = 1

for column in categorical_cols:
    if plotnumber <= len(categorical_cols):
        plt.subplot(3, 5, plotnumber)
        sns.countplot(x=column, data=df, hue=column, palette='rocket', legend=False)
        plt.xlabel(column)
        plt.title(f"Count Plot for {column}")
        plotnumber += 1

plt.tight_layout()
plt.show()


# In[41]:


# Create the countplot
sns.countplot(x=df["ethnicity"])
plt.title("Count Plot for Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Count")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='right')  # Rotate labels by 45 degrees and align them to the right

plt.show()


# In[42]:


# Set figure size to provide more space
plt.figure(figsize=(14, 6))  # Adjust the width (12) to widen the space

# Create the countplot
sns.countplot(x=df["contry_of_res"])
plt.title("Count Plot for contry_of_res")
plt.xlabel("Country of Residence")
plt.ylabel("Count")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='center')  # Rotate labels by 90 degrees and align them to the center

plt.show()


# In[43]:


# countplot for target column (Class/ASD)
sns.countplot(x=df["Class/ASD"])
plt.title("Count Plot for Class/ASD")
plt.xlabel("Class/ASD")
plt.ylabel("Count")
plt.show()


# In[44]:


import matplotlib.pyplot as plt
import pandas as pd

# Correcting the misspelled column name
categorical_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'gender', 'jundice', 'austim',
                    'used_app_before', 'relation']

# Define target
target = 'Class/ASD'

# Define subplot grid size
num_cols = 5
num_rows = 3  # Adjusted since there are 15 categorical features

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 3), constrained_layout=True)
axes = axes.flatten()

# Plot each categorical feature
for i, cat in enumerate(categorical_cols):
    ax = axes[i]
    df_ct = pd.crosstab(df[cat], df[target])
    df_ct.plot(kind='bar', stacked=True, ax=ax, legend=False)
    ax.set_title(f'{cat} vs {target}')
    ax.set_xlabel(cat)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Add single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', title=target)

plt.show()



# In[45]:


# count the outliers using IQR method
Q1_age = df["age"].quantile(0.25)
Q3_age = df["age"].quantile(0.75)
IQR = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR
upper_bound_age = Q3_age + 1.5 * IQR
age_outliers = df[(df["age"] < lower_bound_age) | (df["age"] > upper_bound_age)]
len(age_outliers)


# In[46]:


# count the outliers using IQR method
Q1_result = df["result"].quantile(0.25)
Q3_result = df["result"].quantile(0.75)
IQR = Q3_result - Q1_result
lower_bound_result = Q1_result - 1.5 * IQR
upper_bound_result = Q3_result + 1.5 * IQR
result_outliers = df[(df["result"] < lower_bound_result) | (df["result"] > upper_bound_result)]
len(result_outliers)


# In[47]:


# Capping the outliers in both columns at once
df["age"] = df["age"].clip(lower=lower_bound_age, upper=upper_bound_age)
df["result"] = df["result"].clip(lower=lower_bound_result, upper=upper_bound_result)


# In[48]:


numeric_cols = ['age', 'result']

# Set the figure size
plt.figure(figsize=(12, 4))

# Calculate the number of rows and columns for the subplots
num_plots = len(numeric_cols)
num_cols = 3  # Since you have 3 numerical columns
num_rows = (num_plots // num_cols) + (1 if num_plots % num_cols != 0 else 0)

# Plot boxplots dynamically based on the number of features
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")

# Adjust layout
plt.tight_layout()
plt.show()


# In[49]:


# Function to conduct chi square test between categorical feature and target feature
cat_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'gender', 'jundice', 'austim',
                    'used_app_before', 'relation']
def chi_sq_test(ct):
    # input crosstab of 2 categorical variables
    stat, p, dof, expected = chi2_contingency(ct)

    # interpret p-value
    alpha = 0.05
    print("p value is " + str(p))
    if p <= alpha:
        print('Both variables are Dependent (reject H0)')
    else:
        print('Both variables are Independent (H0 holds true)\n')


# Function to perform chi-square test between categorical feature and target variable (no plotting)
def cat_col_test(df, cat_colname, target_colname):
    print(f"Column name - {cat_colname}")
    ct = pd.crosstab(df[cat_colname], df[target_colname])
    chi_sq_test(ct)

# Run chi-square test on each categorical column
for c in cat_cols:
    cat_col_test(df, c, 'Class/ASD')


# In[50]:


# List of independent features (p-value > 0.05)
independent_features = ['gender', 'used_app_before', 'relation']

# Remove the independent features from the DataFrame
df = df.drop(columns=independent_features)

# Display the cleaned DataFrame
print(df.head())


# In[51]:


# Function to perform ANOVA for all numeric features
def check_dependence(df, target_colname, numeric_cols, alpha=0.05):
    significant_features = []
    for col in numeric_cols:
        # Perform ANOVA for each numerical feature with respect to the categorical target
        groups = [df[df[target_colname] == category][col] for category in df[target_colname].unique()]
        stat, p_value = stats.f_oneway(*groups)

        # Interpret the p-value and store significant features
        print(f"p-value for {col}: {p_value}")
        if p_value <= alpha:
            significant_features.append(col)
            print(f"{col} is dependent on {target_colname} (reject H0)\n")
        else:
            print(f"{col} is independent of {target_colname} (H0 holds true)\n")

    return significant_features

# Example usage:
numeric_cols = ['age','result']
significant_features = check_dependence(df, 'Class/ASD', numeric_cols)


# In[52]:


# one hot encoding for categorical variables with only 2 unique values
df['jundice'] = np.where(df['jundice'] == 'yes',1,0)
print(df['jundice'].value_counts())


# In[53]:


# one hot encoding for categorical variables with only 2 unique values
df['austim'] = np.where(df['austim'] == 'yes',1,0)
print(df['austim'].value_counts())


# In[54]:


# Label encoding based on frequency values of categorical variable
dict_ethnicity = dict(zip(df['ethnicity'].value_counts().index, range(1,df['ethnicity'].nunique()+1)))
dict_ethnicity


# In[55]:


df['ethnicity'] = df['ethnicity'].map(dict_ethnicity)
# Check the value counts in the 'ethnicity' column
# print(df['ethnicity'].value_counts())


# In[56]:


country_freq = df['contry_of_res'].value_counts()

# Select the top 16 countries based on frequency (the countries with the highest occurrence)
top_16_countries = country_freq.head(16).index

# Replace countries not in the top 16 with 'Other'
df['contry_of_res'] = df['contry_of_res'].apply(lambda x: x if x in top_16_countries else 'Other')

# Verify the changes
print(df['contry_of_res'].value_counts())


# In[57]:


dict_country = dict(zip(df['contry_of_res'].value_counts().index, range(1,df['contry_of_res'].nunique()+1)))


# In[58]:


df['contry_of_res'] = df['contry_of_res'].map(dict_country)
print(df['contry_of_res'].value_counts())


# In[59]:


X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[61]:


print(y_train.shape)
print(y_test.shape)


# In[62]:


print(y_train.value_counts())


# In[63]:


smote = SMOTE(random_state=42)


# In[64]:


X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.shape)


# In[65]:


print(y_train_smote.value_counts())


# In[66]:


# MODEL TRAINING
# Dictionary of classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Extra Trees": ExtraTreesClassifier(random_state=42)
}



# In[67]:


# dictionary to store the cross validation results
cv_scores = {}
# perform 5-fold cross validation for each model
for model_name, model in models.items():
  print(f"Training {model_name} with default parameters...")
  scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} Cross-Validation Accuracy: {np.mean(scores):.4f}")
  print("-"*50)


# In[68]:


cv_scores


# In[69]:


#Model Evaluation and Comparison (after baseline modeling)
# Dictionary to store test results
test_scores = {}

# Train each model on the full training set and evaluate on the test set
for model_name, model in models.items():
    model.fit(X_train_smote, y_train_smote)  # Train the model
    y_test_pred = model.predict(X_test)     # Predict on test data

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_scores[model_name] = test_accuracy

    # Print test accuracy
    print(f"\n{model_name} Test Accuracy: {test_accuracy:.4f}")

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    print("=" * 50)


# In[70]:


# Initializing models
random_forest = RandomForestClassifier(random_state=42)
catboost_classifier = CatBoostClassifier(random_state=42, verbose=0)  # Set verbose=0 to suppress CatBoost logs
extra_trees_classifier = ExtraTreesClassifier(random_state=42)


# In[71]:


param_grid_rf = {
    "n_estimators": [10,50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

param_grid_catboost = {
    "iterations": [100, 200, 500],
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5],
    "border_count": [32, 50, 100],
    "thread_count": [4, 8, 16],
}

param_grid_extra_trees = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}


# In[73]:


# Perform RandomizedSearchCV for each model
random_search_rf = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid_rf, n_iter=20, cv=5, scoring="accuracy",random_state=42)
random_search_catboost = RandomizedSearchCV(estimator=catboost_classifier, param_distributions=param_grid_catboost, n_iter=20, cv=5, scoring="accuracy", random_state=42)
random_search_extra_trees = RandomizedSearchCV(estimator=extra_trees_classifier, param_distributions=param_grid_extra_trees, n_iter=20, cv=5, scoring="accuracy", random_state=42)

# Fit the models
random_search_rf.fit(X_train_smote, y_train_smote)
random_search_catboost.fit(X_train_smote, y_train_smote)
random_search_extra_trees.fit(X_train_smote, y_train_smote)

# Retrieve the best models from each search
best_rf = random_search_rf.best_estimator_
best_catboost = random_search_catboost.best_estimator_
best_extra_trees = random_search_extra_trees.best_estimator_

# You can print or save the best models and their respective scores if needed

print(f"Best Random Forest: {best_rf}")
print(f"Best Score: {random_search_rf.best_score_}")
print(f"Best CatBoost: {best_catboost}")
print(f"Best Score: {random_search_catboost.best_score_}")
print(f"Best Extra Trees: {best_extra_trees}")
print(f"Best Score: {random_search_extra_trees.best_score_}")


# In[74]:


# Function to evaluate a model on the test data
def evaluate_on_test(model, X_test, y_test, model_name):
    print(f"---------------- {model_name} ---------------------")
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\n")

# Evaluate Random Forest
evaluate_on_test(best_rf, X_test, y_test, "Random Forest")

# Evaluate CatBoost
evaluate_on_test(best_catboost, X_test, y_test, "CatBoost")

# Evaluate Extra Trees
evaluate_on_test(best_extra_trees, X_test, y_test, "Extra Trees")


# In[75]:


# Function to evaluate a model on the test data
def evaluate_on_test(model, X_test, y_test, model_name):
    print(f"---------------- {model_name} ---------------------")
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\n")

# Evaluate Random Forest
evaluate_on_test(best_rf, X_test, y_test, "Random Forest")

# Evaluate CatBoost
evaluate_on_test(best_catboost, X_test, y_test, "CatBoost")

# Evaluate Extra Trees
evaluate_on_test(best_extra_trees, X_test, y_test, "Extra Trees")


# In[76]:


# Creating the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('cat', best_catboost),
        ('extraTree', best_extra_trees)
    ],
    voting='soft'
)


# In[77]:


# Train the voting classifier
voting_clf.fit(X_train_smote, y_train_smote)

# Predict on test data
y_test_pred = voting_clf.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_test_pred)
print("Voting Classifier Accuracy on Test Data:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Display confusion matrix and classification report
print("\nConfusion Matrix:")
cm=confusion_matrix(y_test, y_test_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[78]:


# Define base models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Define stacking classifier
stack = StackingClassifier(
    estimators=[('rf', rf), ('gb', gb)],  # Base models
    final_estimator=RandomForestClassifier(random_state=42)  # Meta-model
)


# In[79]:


# Train the stacking classifier
stack.fit(X_train_smote, y_train_smote)

##########################
# Evaluate on the test set
y_pred_stack_train = stack.predict(X_train_smote)
accuracy = accuracy_score( y_train_smote, y_pred_stack_train)
print(f"Stacking Classifier training Accuracy: {accuracy:.4f}")

##########################
# Evaluate on the test set
y_pred_stack = stack.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred_stack)
print(f"Stacking Classifier Test Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_stack))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_stack)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the trained stacking model
import joblib
joblib.dump(stack, 'stacking_model.pkl')
print("Model saved as stacking_model.pkl")


# In[ ]:




