import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the model and scaler
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')

st.title("Titanic Survival Prediction")
st.write("Enter the passenger's details to predict if they would have survived.")

# 2. Collect User Input via Streamlit Widgets
pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid (£)", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

if st.button("Predict Survival"):
    # 3. Process the Inputs to Match Training Data Features
    
    # Sex mapping
    sex_male = 1 if sex == "Male" else 0
    
    # Embarked mapping
    embarked_q = 1 if embarked.startswith("Q") else 0
    embarked_s = 1 if embarked.startswith("S") else 0
    
    # Title mapping
    title_miss = 1 if title == "Miss" else 0
    title_mr = 1 if title == "Mr" else 0
    title_mrs = 1 if title == "Mrs" else 0
    title_rare = 1 if title == "Rare" else 0
    
    # Family Size and IsAlone logic
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Fare log transformation (We used np.log1p during training!)
    fare_log = np.log1p(fare)
    
    # 4. Construct the Feature Array
    # MUST be in the exact order as the features list in training:
    # ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']
    features = np.array([[
        pclass, age, fare_log, family_size, is_alone, 
        sex_male, embarked_q, embarked_s, 
        title_miss, title_mr, title_mrs, title_rare
    ]])
    
    # 5. Scale the features and Predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    # 6. Display Results
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🚢 Prediction: This passenger would have SURVIVED!")
    else:
        st.error("⚓ Prediction: This passenger would NOT have survived.")