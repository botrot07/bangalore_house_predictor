import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load data and trained model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

locations = sorted(data['location'].unique())

st.title("Welcome to Bangalore House Price Predictor")
st.write("Want to predict the price of a new House in Bangalore? Try filling the details below:")

# Input Form
location = st.selectbox("Select the Location:", locations)

col1, col2 = st.columns(2)
with col1:
    bhk = st.number_input("Enter BHK:", min_value=1, max_value=16, value=2, step=1)
    bath = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=16, value=2, step=1)

with col2:
    sqft = st.number_input("Enter Total Square Feet:", min_value=300.0, max_value=50000.0, value=1200.0, step=50.0)

if st.button("Predict Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    # Predict and convert to Lakhs -> Rupees
    prediction = pipe.predict(input_data)[0] * 100000
    
    st.markdown(f"### Prediction: ₹{np.round(prediction, 2):,}")