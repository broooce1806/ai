# ai
ai for predictive maintenance
# Predictive Maintenance with AI (AI4I 2020 Dataset)
# This notebook uses the AI4I 2020 dataset to predict machine failure using a Random Forest classifier.

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

## 2. Load and Inspect the Data
df = pd.read_csv("ai4i2020.csv")
df.head()

## 3. Data Cleaning and Preparation
# Drop unnecessary columns
df_cleaned = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

# Rename columns for easier use
df_cleaned.columns = df_cleaned.columns.str.replace(" ", "_").str.replace("[", "").str.replace("]", "")

# One-hot encode the 'Type' column
df_encoded = pd.get_dummies(df_cleaned, columns=["Type"], drop_first=True)

## 4. Split the Data
X = df_encoded.drop("Machine_failure", axis=1)
y = df_encoded["Machine_failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

## 5. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## 6. Evaluate the Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

roc_score = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", roc_score)

## 7. Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=["Healthy", "Fail"], yticklabels=["Healthy", "Fail"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
#| Step                       | Purpose                                       |
#| -------------------------- | --------------------------------------------- |
#| `read_csv()`               | Load real-world data                          |
#| `drop()`                   | Remove irrelevant columns                     |
#| `get_dummies()`            | Convert text into machine-readable numbers    |
#| `train_test_split()`       | Prepare data for training/testing             |
#| `RandomForestClassifier()` | Build a strong predictive model               |
#| `fit()`                    | Train the model                               |
#| `predict()`                | Make predictions                              |
#| `classification_report()`  | Understand model quality                      |
#| `heatmap()`                | Visually understand success and failure cases |
