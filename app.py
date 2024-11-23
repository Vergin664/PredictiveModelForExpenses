

import streamlit as st
import pickle
import numpy as np
from xgboost import XGBClassifier

# Load the trained model
with open('XGB Final model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the Label Encoder
with open('Encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Title and description of the app
st.title("Income and Expense Prediction App")
st.write("Predict the category of expenses based on user details.")

# User Inputs
age = st.number_input("Age:", min_value=1, max_value=90, value=30, step=1)

workclass = st.selectbox(
    "Workclass:",
    ['Self-emp-inc', 'Private', 'State-gov', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
)
education = st.selectbox(
    "Education:",
    ['Bachelors', 'Some-college', 'Doctorate', 'HS-grad', 'Assoc-voc', 'Masters', 
     '7th-8th', '10th', 'Assoc-acdm', '9th', '11th', 'Prof-school', '12th', 
     '1st-4th', '5th-6th', 'Preschool']
)
education_num = st.slider("Education Num:", min_value=1, max_value=16, value=9)

marital_status = st.selectbox(
    "Marital Status:",
    ['Married-civ-spouse', 'Never-married', 'Separated', 'Divorced', 'Widowed', 
     'Married-spouse-absent', 'Married-AF-spouse']
)
occupation = st.selectbox(
    "Occupation:",
    ['Exec-managerial', 'Other-service', 'Prof-specialty', 'Adm-clerical', 'Machine-op-inspct', 
     'Craft-repair', 'Sales', 'Transport-moving', 'Handlers-cleaners', 'Tech-support',  
     'Priv-house-serv', 'Farming-fishing', 'Protective-serv', 'Armed-Forces']
)
relationship = st.selectbox(
    "Relationship:",
    ['Husband', 'Own-child', 'Wife', 'Not-in-family', 'Unmarried', 'Other-relative']
)
race = st.selectbox(
    "Race:",
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
)
sex = st.selectbox("Sex:", ['Male', 'Female'])

capital_gain = st.number_input("Capital Gain:", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss:", min_value=0, value=0)
hours_per_week = st.slider("Hours Per Week:", min_value=1, max_value=100, value=40)

native_country = st.text_input('Enter your native country')

# Encoding categorical variables
try:
    native_country_encoded = label_encoder.transform([native_country])[0]
    occupation_encoded = label_encoder.transform([occupation])[0]
    workclass_encoded = label_encoder.transform([workclass])[0]
    relationship_encoded = label_encoder.transform([relationship])[0]
    race_encoded = label_encoder.transform([race])[0]
    sex_encoded = 1 if sex == 'Male' else 0
    marital_status_encoded = 1 if marital_status.startswith("Married") else 0
    education_encoded = label_encoder.transform([education])[0]
except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# Preparing the input for prediction
input_data = np.array([
    age, education_num, sex_encoded, capital_gain, capital_loss, hours_per_week,
    native_country_encoded, occupation_encoded, workclass_encoded, relationship_encoded,
    race_encoded, marital_status_encoded, education_encoded
]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted expense category is: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
