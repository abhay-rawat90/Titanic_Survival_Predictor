import streamlit as st
import joblib
import numpy as np

model = joblib.load('titanic_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')

st.title("Titanic Survival Prediction")
st.write("Enter the passenger's details below to predict their survival.")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=32.0)
sex = st.selectbox("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

if st.button("Predict Survival"):
    
    sex_male = 1 if sex == "Male" else 0
    
    embarked_q = 1 if embarked.startswith("Q") else 0
    embarked_s = 1 if embarked.startswith("S") else 0
    
    features_array = np.array([[
        pclass, 
        age, 
        sibsp, 
        parch, 
        fare, 
        sex_male, 
        embarked_q, 
        embarked_s
    ]])
    
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.success("Prediction: This passenger would have SURVIVED!")
    else:
        st.error("Prediction: This passenger would NOT have survived.")
