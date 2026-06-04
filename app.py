import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load Dataset
data = pd.read_csv("Student.csv")

# Preprocess Data
data["Result"] = data["Result"].astype(str).str.strip().str.upper()

encoder = LabelEncoder()
data["Result"] = encoder.fit_transform(data["Result"])

# Features and Target
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Result"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Student Performance Prediction System")

st.write("Enter student details to predict PASS or FAIL")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Result"):
    new_student = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=["Study_Hours", "Attendance", "Previous_Score"]
    )

    prediction = model.predict(new_student)

    if prediction[0] == 1:
        st.success("PASS ✅")
    else:
        st.error("FAIL ❌") 
