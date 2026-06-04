# Student Performance Prediction System

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("Student.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nStatistical Summary:")
print(data.describe())

# Clean Result Column
data["Result"] = data["Result"].astype(str).str.strip().str.upper()

# Convert PASS/FAIL to Numeric Values
encoder = LabelEncoder()
data["Result"] = encoder.fit_transform(data["Result"])

# Features and Target Variable
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Result"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with New Student Data
study_hours = 6
attendance = 85
previous_score = 70

new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score]],
    columns=["Study_Hours", "Attendance", "Previous_Score"]
)

prediction = model.predict(new_student)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL") 
