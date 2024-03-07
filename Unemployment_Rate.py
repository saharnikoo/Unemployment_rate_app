


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('best_random_forest_model.pkl')

# Define the input fields
st.title("Kenya Unemployment Rate Prediction App")

# Add a rainbow-colored divider using Markdown
st.markdown("<hr style='border: 2px solid green; background-color: red;'>", unsafe_allow_html=True)
st.write("App by Sahar Nikoo")
st.write("Email Address: nikoo_sahar@yahoo.com")
st.write("This app predicts the Total Unemployment in Kenya.")



#(using st.slider)

Real_GDP_Ksh= st.slider("Real_GDP_Ksh", min_value=0, max_value=1000000)
Population_Growth= st.slider("Population Growth", min_value=-5.0, max_value=5.0, value=-5.0)
Female_Labor_Participation= st.slider("Female Labor Participation", min_value=0.0, max_value=100.0, value=00.0)
Male_Labor_Participation= st.slider("Male Labor Participation", min_value=0.0, max_value=100.0, value=00.0)
Education_Expenditure_Ksh = st.slider("Education_Expenditure_Ksh", min_value=40000000000.0, max_value=100000000000.0, value=40000000000.0)
Inflation= st.slider("Inflation", min_value=0.0, max_value=50.0, value=00.0)
Dollar_Rate = st.slider("Male Labor ParMale_Labor_Participationticipation", min_value=0.0, max_value=150.0, value=00.0)
Labor_Total_Population_Ratio= st.slider("Labor_Total_Population_Ratio", min_value=0.0, max_value=1.0, value=0.0)
Urban_Population_Growth_Income_Per_Capita_Growth_Ratio= st.slider("Urban_Population_Growth_Income_Per_Capita_Growth_Ratio", min_value=0.0, max_value=2.0, value=00.0)


# Create a feature vector from user inputs
feature_vector = np.array([Real_GDP_Ksh, Population_Growth, Female_Labor_Participation, Male_Labor_Participation, Education_Expenditure_Ksh, Inflation, Dollar_Rate, Labor_Total_Population_Ratio, Urban_Population_Growth_Income_Per_Capita_Growth_Ratio]).reshape(1, -1)



# Make predictions
if st.button("Predict"):
    prediction = model.predict(feature_vector)
    st.success(f"Predicted Total Unemployment: {prediction[0]:.2f}")

