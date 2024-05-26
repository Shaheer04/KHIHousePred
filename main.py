import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Music Recommendation System")

try:
    with open('finalized_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading pickle file: {e}")

features = pd.read_csv('features.csv')

def prediction(location,sqft,bedrooms,baths):
    loc_index = np.where(features.columns==location)[0][0]

    x = np.zeros(len(features.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
      
    result = model.predict([x])[0] / 1000000
    print(result)
    inflation_rate = 19.87
    multiplier = 1 + (inflation_rate / 100)
    adjusted_price = result * multiplier

    return adjusted_price

st.title("Karachi House Price Prediction")
st.write("This is a simple web app to predict house prices in Karachi, Pakistan.")

location = st.selectbox("Select Location", features.columns[3:])
sqft = st.number_input("Square Feet")
bedrooms = st.slider("Bedrooms" , max_value=8)
baths = st.slider("Bathrooms", max_value=6)

if st.button("Predict Price"):
    price = str(int(prediction(location, sqft, bedrooms, baths)))
    st.write(f"The estimated price of the house is: {price} Million PKR")




