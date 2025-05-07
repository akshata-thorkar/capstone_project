import os
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

# Get the working directory of the script
#working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained heart disease model
# model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
# scaler = pickle.load(open(f'{working_dir}/saved_models/scaler.pkl', 'rb'))

model = pickle.load(open('saved_models/heart_disease_model.sav', 'rb'))
scaler = pickle.load(open('saved_models/scaler.pkl', 'rb'))
# Title
st.title('Heart Disease Prediction using Machine Learning')

# Input layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=0)

with col2:
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

with col3:
    cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])

with col1:
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0)

with col2:
    chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=0)

with col3:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1])

with col1:
    restecg = st.selectbox('Resting ECG Results (restecg)', options=[0, 1, 2])

with col2:
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0)

with col3:
    exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1])

with col1:
    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', format="%.1f")

with col2:
    slope = st.selectbox('Slope of Peak Exercise ST Segment (slope)', options=[0, 1, 2])

with col3:
    ca = st.selectbox('Number of Major Vessels Colored (ca)', options=[0, 1, 2, 3, 4])

with col1:
    thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])

# Prediction
if st.button('Predict Heart Disease'):
    user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, 
                   thalach, exang, oldpeak, slope, ca, thal]]

    user_input_scaled = scaler.transform(user_input)  # ⚠️ Apply scaling!
    prediction = model.predict(user_input_scaled)

    if prediction[0] == 1:
        st.error('⚠️ The person is likely to have heart disease.')
    else:
        st.success('✅ The person is unlikely to have heart disease.')
