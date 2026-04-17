import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("C:/Users/patel/Desktop/Guddy/Machine Learning/Projects/Project(Heart analysis)/KNN_Heart.pkl")
scaler = joblib.load("C:/Users/patel/Desktop/Guddy/Machine Learning/Projects/Project(Heart analysis)/Scaler_Heart.pkl")
Expected_columns = joblib.load("C:/Users/patel/Desktop/Guddy/Machine Learning/Projects/Project(Heart analysis)/Columns_Heart.pkl")


st.title("Heart stroke Prediction 💖")
st.markdown("Provide the following details to check your heart stroke risk:")

# Collect user input
Age = st.slider("Age", 18, 100, 40)
Sex = st.selectbox("SEX", ["M", "F"])
Chest_Pain_Type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
Resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholesterol = st.number_input("Cholesterol (mg/dl)", 80, 600, 200)
Fasting_BS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
Resting_ECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
Max_HR = st.slider("Max Heart Rate", 60, 220, 150)
Exercise_Angina = st.selectbox("Exercise-Induces Angina", ["Y", "N"])
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
St_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# When Predict is clicked
if st.button("Predict"):
    # Create a raw input dictionary
    raw_input = {
        'Age': Age,
        'RestingBP': Resting_BP,
        'Cholesterol': Cholesterol,
        'FastingBS': Fasting_BS,
        'MaxHR': Max_HR,
        'Oldpeak': Oldpeak,
        'Sex_' + Sex: 1,
        'ChestPainType_' + Chest_Pain_Type: 1,
        'RestingECG_' + Resting_ECG: 1,
        'ExerciseAngina_' + Exercise_Angina: 1,
        'ST_Slope_' + St_Slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in Expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[Expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
