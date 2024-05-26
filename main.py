import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Karachi House Price Prediction")

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

def sqft_to_gaz(sqft):
    return sqft / 9

st.title("Karachi House Price Prediction")
st.subheader("A simple web app to predict house prices in Karachi, Pakistan.")
st.warning("""The house price predictions provided by this project are intended solely for informational purposes and are not guaranteed to be accurate. These predictions are generated using a machine learning model trained on historical data sourced from Kaggle, which has been adjusted to reflect an inflation rate of 19.87%. """)



tab1 ,tab2 = st.tabs(["Predict Price", "About"])
with tab1:
    location = st.selectbox("Select Location", features.columns[3:])
    sqft = st.number_input("Square Feet")
    gaz_sqft = str(int(sqft_to_gaz(sqft)))
    st.info(f"Square Feet converted to yards: {gaz_sqft}")
    bedrooms = st.slider("Bedrooms" , max_value=8)
    baths = st.slider("Bathrooms", max_value=6)

    if st.button("Predict Price"):
        price = str(int(prediction(location, sqft, bedrooms, baths)))
        st.success(f"The estimated price of the house is: {price} Million PKR", icon="🏠")

with tab2:
    st.title("About")
    st.link_button("GitHub", "https://github.com/Shaheer04")
    st.link_button("LinkedIn Profile", "https://www.linkedin.com/in/shaheerjamal/")



