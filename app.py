
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load the pre-trained model and encoder
with open('XGB Final model.pkl', 'rb') as file:
    final_model = pickle.load(file)

with open('Encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


# Define the prediction function
def prediction(input_data):
    # Convert input_data to numpy array for prediction
    input_data = np.array(input_data, dtype='object')

    # Get prediction probability for class 1 (income >50K)
    pred = model.predict_proba(input_data)[:, 1][0]

    # Check the prediction and return the appropriate message
    if pred > 0.5:
        return f'This individual is more likely to earn > 50K: Probability = {round(pred * 100, 2)}%'
    else:
        return f'This individual is less likely to earn > 50K: Probability = {round(pred * 100, 2)}%'


# Streamlit UI
st.title("Income Prediction Model")

# Input fields for the features
age = st.slider('Select Age', 18, 100, 30)  # Age slider
education_num = st.slider('Select Education Level (Numeric)', 1, 16, 12)  # Education num slider
capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=5000)
capital_loss = st.number_input('Capital Loss', min_value=0, max_value=99999, value=0)
hours_per_week = st.slider('Hours Per Week', 1, 100, 40)  # Hours per week slider

# Select boxes for categorical variables
sex = st.selectbox('Select Sex', ['Male', 'Female'])
native_country = st.text_input('Enter Native Country')

occupation = st.selectbox('Select Occupation', ['Exec-managerial', 'Other-service', 'Prof-specialty', 'Adm-clerical', 'Machine-op-inspct', 
     'Craft-repair', 'Sales', 'Transport-moving', 'Handlers-cleaners', 'Tech-support', 
     'Priv-house-serv', 'Farming-fishing', 'Protective-serv', 'Armed-Forces'])

workclass = st.selectbox('Select Workclass', ['Self-emp-inc', 'Private', 'State-gov', 
                                              'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'Without-pay', 'Never-worked'])

relationship = st.selectbox('Select Relationship', ['Husband', 'Own-child', 'Wife', 'Not-in-family', 'Unmarried', 'Other-relative'])

race = st.selectbox('Select Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

marital_status = st.selectbox('Select Marital Status', ['Married-civ-spouse', 'Never-married', 'Separated', 
                                                        'Divorced', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

education = st.selectbox('Education:', ['Bachelors', 'Some-college', 'Doctorate', 'HS-grad', 'Assoc-voc', 
                                        'Masters', '7th-8th', '10th', 'Assoc-acdm', '9th', '11th', 'Prof-school',
                                        '12th', '1st-4th', '5th-6th', 'Preschool'])

# Encode categorical variables
sex_encoded = label_encoder.transform([sex])[0]
native_country_encoded = label_encoder.transform([native_country])[0]
occupation_encoded = label_encoder.transform([occupation])[0]
workclass_encoded = label_encoder.transform([workclass])[0]
relationship_encoded = label_encoder.transform([relationship])[0]
race_encoded = label_encoder.transform([race])[0]
marital_status_encoded = label_encoder.transform([marital_status])[0]
education_encoded = label_encoder.transform([education])[0]

# Prepare input data for the model
input_data = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'sex': [sex_encoded],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country1': [native_country_encoded],
    'occupation1': [occupation_encoded],
    'workclass1': [workclass_encoded],
    'relationship1': [relationship_encoded],
    'race1': [race_encoded],
    'marital-status2': [marital_status_encoded],
    'education1': [education_encoded] 
})


 if st.button('Predict'):
    response = prediction(input_data)
    st.success(response)

if __name__ == '__main__':
    main()
