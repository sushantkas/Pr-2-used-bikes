import streamlit as st
import pandas as pd
import numpy as np
import importlib
import preprocessing_d
importlib.reload(preprocessing_d)
import preprocessing_d
import seaborn as sns
import pickle

@st.cache_data
def basic_info():
    ref_data= pd.read_csv("bikes_EDA_Cleaned.csv")

    X_columns=['model_name', 'model_year', 'kms_driven', 'owner', 'location',
       '    mileage', 'power', 'brand', 'cc']
    
    model_name=ref_data['model_name'].unique()
    with open("categorical_encoders_freq.pkl", "rb") as f:
            freq = pickle.load(f)
    return ref_data, X_columns, model_name ,freq

ref_data, X_columns, model_name, freq = basic_info()


st.set_page_config(layout="wide")
st.header('Welcome to Bike price predictor 🏍️💨')

st.subheader("Bike Price will be Estimated Based on several Factors")


on = st.toggle("Do you have any CSV File ? Click here to Upload and predict")

if on:
    st.write("Feature activated!")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        option_model_name= st.selectbox(
            "select Bike Model",
            (model_name))
        option_model_year = st.selectbox('Model_year', range(1950,2025,1))
        option_kms_driven = st.slider('KMS Driven', 0, 1000000, 50000)
        option_owner = st.pills('Owner', options=[1,2,3,4], selection_mode="single")
        option_location = st.selectbox('Bike Located in', freq['location'].keys())
        
    with col2:
        option_mileage = st.slider('mileage', 5, 100, step=5)
        option_power = st.slider('power', 5, 250, 5)
        option_cc = st.slider('Bike CC', 100, 2000, step=5)
        option_brand = st.selectbox('Bike Brand', freq['brand'].keys())
    with col3:
        st.write('Model Name: ', option_model_name)
        st.write('Model Year: ', option_model_year)
        st.write('KMS Driven: ', option_kms_driven)
        st.write('Owner: ', option_owner)
        st.write('Location: ', option_location)
        st.write('Mileage: ', option_mileage)
        st.write('Power: ', option_power)
        st.write('CC: ', option_cc)
        st.write('Brand: ', option_brand)
        # Changing Into DataFrame
        df = {
            'model_name': [option_model_name],
            'model_year': [option_model_year],
            'kms_driven': [option_kms_driven],
            'owner': [option_owner],    
            'location': [option_location],
            'mileage': [option_mileage],
            'power': [option_power],
            'brand': [option_brand],
            'cc': [option_cc]}
        df=pd.DataFrame(df)
        from preprocessing_d import pipelines_ml
        
        st.write(f"Estimated Bike Price:  ₹{int(pipelines_ml(df).predict()["price"].values[0])}")
