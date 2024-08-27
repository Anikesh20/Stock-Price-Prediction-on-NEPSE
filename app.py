import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
from tensorflow.keras.models import load_model

# Directory where all CSV files are stored
directory = 'E:/BCA/PROJECT/Data/Data(Stock Trading)'

# Function to extract company names from CSV filenames
def get_company_names(directory):
    company_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Assuming the filename is structured as 'company_name.csv'
            company_name = filename.replace('.csv', '')
            company_names.append(company_name)
    return sorted(company_names)

# Load the trained model and scaler
def load_best_model_and_scaler():
    try:
        # Attempt to load the LSTM model first
        model = load_model('lstm_stock_price_predictor.h5')
        model_type = 'LSTM'
    except:
        # If LSTM model isn't found, fall back to traditional models
        model = load('LinearRegression_stock_price_predictor.joblib')
        model_type = 'Traditional'
    scaler = load('scaler.joblib')
    return model, scaler, model_type

# Load the best model and scaler
model, scaler, model_type = load_best_model_and_scaler()

# Define the feature order used during training
feature_order = [
    'Prev_Close',
    'Price_Change',
    'HIGH PRICE',
    'LOW PRICE',
    'TOTAL TRADED QUANTITY',
    'TOTAL TRADED VALUE',
    'TOTAL TRADES'
]

# Function to predict future price
def predict_price(input_data, model, model_type):
    # Ensure input_data columns match the feature order
    input_data = input_data[feature_order]
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    if model_type == 'LSTM':
        # Reshape data for LSTM
        input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
        prediction = model.predict(input_data_scaled)
    else:
        # Predict using the trained model
        prediction = model.predict(input_data_scaled)
    
    return prediction[0]

# Function to load stock data and plot the index chart
def plot_stock_chart(stock_name):
    file_path = os.path.join(directory, f'{stock_name}.csv')
    stock_data = pd.read_csv(file_path)
    
    # Assuming 'BUSINESS DATE' and 'CLOSE PRICE' are in the CSV
    stock_data['BUSINESS DATE'] = pd.to_datetime(stock_data['BUSINESS DATE'])
    stock_data.sort_values('BUSINESS DATE', inplace=True)
    
    st.line_chart(stock_data.set_index('BUSINESS DATE')['CLOSE PRICE'])
    
    # Display the latest 5 rows
    st.write("Latest 5 data points:")
    st.dataframe(stock_data.tail(5))

# Streamlit app
st.title("NEPSE Stock Price Predictor")

# Get the list of company names from the directory
company_names = get_company_names(directory)

# Dropdown menu for stock selection
selected_stock = st.selectbox("Select the stock", company_names)

# Display the stock's index chart and the latest 5 data points
if selected_stock:
    plot_stock_chart(selected_stock)

# Input fields for the required features
prev_close = st.number_input("Previous Close Price", min_value=0.0, format="%.2f")
high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
total_traded_quantity = st.number_input("Total Traded Quantity", min_value=0, format="%d")

# User-friendly input for TOTAL TRADED VALUE
total_traded_value_input = st.text_input("Total Traded Value (e.g., 104,000,000 or 1.04E+08)")

# Convert the input to a float, handling both formats
try:
    total_traded_value = float(total_traded_value_input.replace(",", ""))
except ValueError:
    st.error("Please enter a valid number for Total Traded Value.")
    total_traded_value = 0.0

total_trades = st.number_input("Total Trades", min_value=0, format="%d")

# Automatically calculate the Price Change
price_change = ((prev_close - low_price) / low_price) * 100 if low_price != 0 else 0

# Convert inputs to a DataFrame with the correct feature order
input_data = pd.DataFrame({
    'Prev_Close': [prev_close],
    'Price_Change': [price_change],
    'HIGH PRICE': [high_price],
    'LOW PRICE': [low_price],
    'TOTAL TRADES': [total_trades],
    'TOTAL TRADED QUANTITY': [total_traded_quantity],
    'TOTAL TRADED VALUE': [total_traded_value]
})

# Button to trigger the prediction
if st.button("Predict Future Price"):
    try:
        # Predict the future price
        predicted_price = predict_price(input_data, model, model_type)
        st.write(f"The predicted future price for {selected_stock} is: {predicted_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
