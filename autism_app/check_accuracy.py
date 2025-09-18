import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import scipy.stats as stats
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv("C://Users//Madhura Sharan//OneDrive//Desktop//Major Project//ASDD//Autism Screening.csv")

# Data preprocessing
df["age"] = df["age"].replace('?', np.nan)
df["age"] = df["age"].astype(float)
df["age"] = df["age"].fillna(df["age"].median())
df["age"] = df["age"].astype(int)

df['ethnicity'] = df['ethnicity'].replace('?', 'Unknown')
most_frequent_relation = df['relation'].mode()[0]
df['relation'] = df['relation'].replace('?', most_frequent_relation)

mapping = {
    "AmericanSamoa": "United States",
    "Hong Kong": "China"
}
df["contry_of_res"] = df["contry_of_res"].replace(mapping)

# Drop columns
df = df.drop(columns=["age_desc"])

# Remove independent features
independent_features = ['gender', 'used_app_before', 'relation']
df = df.drop(columns=independent_features)

# Encoding categorical variables
df['jundice'] = np.where(df['jundice'] == 'yes',1,0)
df['austim'] = np.where(df['austim'] == 'yes',1,0)

dict_ethnicity = dict(zip(df['ethnicity'].value_counts().index, range(1,df['ethnicity'].nunique()+1)))
df['ethnicity'] = df['ethnicity'].map(dict_ethnicity)

country_freq = df['contry_of_res'].value_counts()
top_16_countries = country_freq.head(16).index
df['contry_of_res'] = df['contry_of_res'].apply(lambda x: x if x in top_16_countries else 'Other')
dict_country = dict(zip(df['contry_of_res'].value_counts().index, range(1,df['contry_of_res'].nunique()+1)))
df['contry_of_res'] = df['contry_of_res'].map(dict_country)

# Features and target
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

# Train-test split (using different random_state to check generalization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# SMOTE for balancing
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Load trained stacking model
model = joblib.load("autism_app/stacking_model.pkl")

# Predict on test data
y_test_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Stacking Classifier Test Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
cr = classification_report(y_test, y_test_pred)
print("Classification Report:")
print(cr)

# Cross-validation to check for overfitting
print("\nCross-Validation Scores:")
# Define base models for stacking
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42)

# Define stacking classifier
stack_cv = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('cat', cat)],
    final_estimator=RandomForestClassifier(random_state=42)
)

# Perform cross-validation
cv_scores = cross_val_score(stack_cv, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print("Individual CV Scores:", cv_scores)
