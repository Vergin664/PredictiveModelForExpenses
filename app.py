
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
import pickle

# Load the saved model and encoder
with open('XGB Final model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Prediction function
def prediction(input_data):
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame(input_data, columns=[
            'age', 'workclass', 'education', 'education-num', 'marital-status', 
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
            'capital-loss', 'hours-per-week', 'native-country'
        ])


        # Apply encoding using the loaded encoder (OneHot or Label Encoding)
        encoded_data = encoder.transform(input_df)

        # Predict the probability of the "Expense" class
        pred = model.predict_proba(encoded_data)[:, 1][0]

        # Return a formatted prediction message
        if pred > 0.5:
            return f"This person is more likely to have an expense >50K: Probability = {round(pred * 100, 2)}%"
        else:
            return f"This person is more likely to have an expense <=50K: Probability = {round(pred * 100, 2)}%"

    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Main function to build the Streamlit app
def main():
    st.title('Income Expense Prediction App')
    

    # User inputs
    age = st.number_input('Age:', min_value=0, max_value=120, value=39, step=1)
    workclass = st.selectbox('Workclass:', ['Self-emp-inc', 'Private', 'State-gov', 'Local-gov', 'Self-emp-not-inc', 'Unknown', 'Federal-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox('Education:', ['Bachelors', 'Some-college', 'Doctorate', 'HS-grad', 'Assoc-voc', 'Masters', '7th-8th', '10th', 'Assoc-acdm', '9th', '11th', 'Prof-school', '12th', '1st-4th', '5th-6th', 'Preschool'])
    education_num = st.number_input('Education Number:', min_value=1, max_value=20, value=13, step=1)
    marital_status = st.selectbox('Marital Status:', ['Married-civ-spouse', 'Never-married', 'Separated', 'Divorced', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox('Occupation:', ['Exec-managerial', 'Other-service', 'Prof-specialty', 'Adm-clerical', 'Machine-op-inspct', 'Craft-repair', 'Sales', 'Transport-moving', 'Handlers-cleaners', 'Tech-support', 'Unknown', 'Priv-house-serv', 'Farming-fishing', 'Protective-serv', 'Armed-Forces'])
    relationship = st.selectbox('Relationship:', ['Husband', 'Own-child', 'Wife', 'Not-in-family', 'Unmarried', 'Other-relative'])
    race = st.selectbox('Race:', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex = st.selectbox('Sex:', ['Male', 'Female'])
    capital_gain = st.number_input('Capital Gain:', min_value=0, value=15024, step=1)
    capital_loss = st.number_input('Capital Loss:', min_value=0, value=0, step=1)
    hours_per_week = st.number_input('Hours per Week:', min_value=0, max_value=168, value=50, step=1)
    native_country = st.selectbox('Native Country:', ['United-States', 'Germany', 'Japan', 'Yugoslavia', 'Unknown', 'India', 'Canada', 'Iran', 'Mexico', 'Taiwan', 'China', 'Jamaica', 'Dominican-Republic', 'England', 'Cuba', 'Philippines', 'El-Salvador', 'Italy', 'Poland', 'Thailand', 'Peru'])

    # Prepare input list for prediction
    input_list = [[
        age, workclass, education, education_num, marital_status, 
        occupation, relationship, race, sex, capital_gain, 
        capital_loss, hours_per_week, native_country
    ]]

    # Predict and display results
    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()
