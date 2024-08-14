# Logistic Regression on Adult Income Data
# Dataset: https://archive.ics.uci.edu/dataset/2/adult
# Kaggle Code: https://www.kaggle.com/code/kartik1trivedi/income-predictor-logistic-regression

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report, RocCurveDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Data Reading
# Training Data
df_train = pd.read_csv(
    r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\adult\adult.data",
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
)
df_train['type'] = 'train'

# Test Data
df_test = pd.read_csv(
    r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\adult\adult.test",
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],
    skiprows=1
)
df_test['type'] = 'test'

# Combine Training and Test Data
df = pd.concat([df_train, df_test], ignore_index=True)

# Check for Missing Values and Data Info
print("Missing Values in Each Column:")
print(df.isnull().sum())
print("\nData Info:")
print(df.info())

# Display Unique Values for Each Column
print("\nUnique Values in Each Column:")
for column in df.columns:
    print(f"Column: {column}")
    print(df[column].unique())

# Clean the Income Column
print("\nCleaned Income Column:")
df['income'] = df['income'].str.strip().str.rstrip('.')
print(df['income'].unique())

# Replace '?' with NaN
df.replace(" ?", np.nan, inplace=True)

# Impute Missing Values with Most Frequent Values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Check for Missing Values After Imputation
print("\nMissing Values After Imputation:")
print(df_imputed.isnull().sum())

# Encode Categorical Features
label_encoder = LabelEncoder()
categorical_columns = df_imputed.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df_imputed[col] = label_encoder.fit_transform(df_imputed[col])

# Prepare Features and Target Variable
X = df_imputed.drop(['income', 'type', 'relationship'], axis=1)
y = df_imputed['income']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
print("\nModel Evaluation:")
print('Test Score:', model.score(X_test, y_test))
print('Train Score:', model.score(X_train, y_train))

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ROC Curve
print("\nROC Curve:")
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()

# Feature Correlation Heatmap
print("\nFeature Correlation Heatmap:")
plt.figure(figsize=(25, 20))
sns.heatmap(df_imputed.corr(), annot=True)
plt.show()
