import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("placement_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Placement Predictor")
st.write("Predict whether a student is likely to get placed using Machine Learning.")

ssc_p = st.number_input("SSC Percentage", 0.0, 100.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0)
etest_p = st.number_input("Employability Test Percentage", 0.0, 100.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0)

gender = st.selectbox("Gender", ["M", "F"])
workex = st.selectbox("Work Experience", ["Yes", "No"])
specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])
hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])

if st.button("Predict Placement"):

    academic_average = (ssc_p + hsc_p + degree_p) / 3

    input_data = {}

    for col in model_columns:
        input_data[col] = 0

    input_data["ssc_p"] = ssc_p
    input_data["hsc_p"] = hsc_p
    input_data["degree_p"] = degree_p
    input_data["etest_p"] = etest_p
    input_data["mba_p"] = mba_p
    input_data["academic_average"] = academic_average

    if gender == "M":
        input_data["gender_M"] = 1

    if hsc_s == "Commerce":
        input_data["hsc_s_Commerce"] = 1

    if hsc_s == "Science":
        input_data["hsc_s_Science"] = 1

    if degree_t == "Others":
        input_data["degree_t_Others"] = 1

    if degree_t == "Sci&Tech":
        input_data["degree_t_Sci&Tech"] = 1

    if workex == "Yes":
        input_data["workex_Yes"] = 1

    if specialisation == "Mkt&HR":
        input_data["specialisation_Mkt&HR"] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.success("🎉 Student is Likely to be Placed")
    else:
        st.error("❌ Student is Less Likely to be Placed")

    st.progress(int(probability))
    st.write(f"Placement Probability: **{probability:.2f}%**")