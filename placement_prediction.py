# ==========================================
# Student Placement Prediction
# Author: Rishik Rai
# ==========================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("All libraries imported successfully!")
# ==========================================
# Load the Dataset
# ==========================================

df = pd.read_csv("Placement_Data_Full_Class.csv")

print("Dataset loaded successfully!\n")
# Display first 5 rows
print(df.head())
# ==========================================
# Explore the Dataset
# ==========================================

print("\n================ Dataset Shape ================")
print(df.shape)

print("\n================ Column Names ================")
print(df.columns)

print("\n================ Data Types ================")
print(df.dtypes)

print("\n================ Dataset Information ================")
df.info()

print("\n================ Missing Values ================")
print(df.isnull().sum())

print("\n================ Statistical Summary ================")
print(df.describe())
# ==========================================
# Data Cleaning
# ==========================================

print("\n================ DATA CLEANING ================\n")

# Remove unnecessary columns
df.drop("salary", axis=1, inplace=True)
df.drop("sl_no", axis=1, inplace=True)

print("Columns after dropping salary and sl_no:\n")
print(df.columns)
# ==========================================
# Check Percentage Columns
# ==========================================

percentage_columns = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]

print("\nChecking percentage columns:\n")

for column in percentage_columns:
    print(f"{column}")
    print(f"Minimum Value : {df[column].min()}")
    print(f"Maximum Value : {df[column].max()}")
    print("-" * 35)
# ==========================================
# Create Images Folder
# ==========================================

os.makedirs("static/images", exist_ok=True)
# ==========================================
# EDA 1 - Placement Status
# ==========================================

plt.figure(figsize=(6,5))

ax = sns.countplot(x="status", data=df)

plt.title("Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

# Add count labels on top of bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig("static/images/placement_status.png")

plt.show()
# ==========================================
# EDA 2 - Placement by Work Experience
# ==========================================

plt.figure(figsize=(7,5))

ax = sns.countplot(
    x="workex",
    hue="status",
    data=df
)

plt.title("Placement Status by Work Experience")
plt.xlabel("Work Experience")
plt.ylabel("Number of Students")

# Add labels on bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig("static/images/placement_by_workex.png")

plt.show()
# ==========================================
# EDA 3 - Placement by Specialisation
# ==========================================

plt.figure(figsize=(7,5))

ax = sns.countplot(
    x="specialisation",
    hue="status",
    data=df
)

plt.title("Placement Status by Specialisation")
plt.xlabel("Specialisation")
plt.ylabel("Number of Students")

# Add labels
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig("static/images/placement_by_specialisation.png")

plt.show()
# ==========================================
# EDA 4 - SSC Percentage vs Placement
# ==========================================

plt.figure(figsize=(7,5))

sns.boxplot(x="status", y="ssc_p", data=df)

plt.title("SSC Percentage by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("SSC Percentage")

plt.tight_layout()

plt.savefig("static/images/ssc_vs_placement.png")

plt.show()
# ==========================================
# EDA 5 - HSC Percentage vs Placement
# ==========================================

plt.figure(figsize=(7,5))

sns.boxplot(x="status", y="hsc_p", data=df)

plt.title("HSC Percentage by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("HSC Percentage")

plt.tight_layout()

plt.savefig("static/images/hsc_vs_placement.png")

plt.show()
# ==========================================
# EDA 6 - Degree Percentage vs Placement
# ==========================================

plt.figure(figsize=(7,5))

sns.boxplot(x="status", y="degree_p", data=df)

plt.title("Degree Percentage by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Degree Percentage")

plt.tight_layout()

plt.savefig("static/images/degree_vs_placement.png")

plt.show()
# ==========================================
# EDA 7 - Correlation Heatmap
# ==========================================

# Create a temporary dataframe
heatmap_df = df.copy()

# Convert target to numeric
heatmap_df["status"] = heatmap_df["status"].map({
    "Placed": 1,
    "Not Placed": 0
})

# Select only numeric columns
numeric_df = heatmap_df.select_dtypes(include=["number"])

plt.figure(figsize=(10,7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("static/images/correlation_heatmap.png")

plt.show()
# ==========================================
# EDA 8 - MBA Percentage vs Placement
# ==========================================

plt.figure(figsize=(7,5))

sns.boxplot(x="status", y="mba_p", data=df)

plt.title("MBA Percentage by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("MBA Percentage")

plt.tight_layout()

plt.savefig("static/images/mba_vs_placement.png")

plt.show()
# ==========================================
# FEATURE ENGINEERING
# ==========================================

model_df = df.copy()

# Create academic average feature
model_df["academic_average"] = (
    model_df["ssc_p"] +
    model_df["hsc_p"] +
    model_df["degree_p"]
) / 3

# Encode target variable
model_df["status"] = model_df["status"].map({
    "Placed": 1,
    "Not Placed": 0
})

# One-Hot Encoding
model_df = pd.get_dummies(
    model_df,
    columns=[
        "gender",
        "hsc_s",
        "degree_t",
        "workex",
        "specialisation"
    ],
    drop_first=True
)

# Remove unused categorical columns
model_df.drop(columns=["ssc_b", "hsc_b"], inplace=True)

print("\n================ FEATURE ENGINEERING ================\n")
print(model_df.head())

print("\nShape :", model_df.shape)

print("\nFinal Columns Used For Model:\n")
print(model_df.columns)

# ==========================================
# MODEL BUILDING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Separate Features and Target
X = model_df.drop("status", axis=1)
y = model_df["status"]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========\n")
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
import joblib

joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model Columns Saved Successfully!")

print("Model Saved Successfully!")

print("\nModel Trained Successfully!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n========== MODEL EVALUATION ==========\n")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))
# ==========================================
# Evaluation Metrics Table
# ==========================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]
})

print("\n========== EVALUATION METRICS ==========\n")
print(results)
# ==========================================
# Confusion Matrix
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("static/images/confusion_matrix.png")

plt.show()
# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========\n")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["Absolute Coefficient"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print(importance)

# ==========================================
# Plot Feature Importance
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Absolute Coefficient",
    y="Feature"
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig("static/images/feature_importance.png")

plt.show()
# ==========================================
# RECOMMENDATIONS
# ==========================================

print("\n================ RECOMMENDATIONS ================\n")

print("1. Students should maintain strong academic performance because higher academic scores are associated with better placement chances.")

print("\n2. Students should gain internship or work experience before placements, as work experience is one of the strongest predictors of getting placed.")

print("\n3. Students should improve aptitude, communication skills, and technical knowledge in addition to academics to increase employability.")

print("\n===============================================")
print(" Student Placement Prediction Project Completed ")
print(" All graphs saved successfully in the Images folder ")
print("===============================================")
import joblib

joblib.dump(model, "placement_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("\nModel Saved Successfully!")