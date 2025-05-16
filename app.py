
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
import logging
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start = "2009-01-01"
end = "2023-01-01"


st.title('📈 Stock Closing Price Prediction')
user_input = st.text_input('Enter Stock Ticker', 'GOOGL')

# Fetch data
df = yf.download(user_input, start=start, end=end)

if df.empty:
    st.error("No data found. Please check the stock ticker symbol.")
    st.stop()

st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# First Plot: Closing price
st.subheader('📊 Closing Price vs Time')
fig1 = plt.figure(figsize=(12, 6))

plt.plot(df['Close'], label='Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Second Plot: With 100 MA
st.subheader('🟢 Closing Price with 100-Day Moving Average')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'r', label='Close Price')
plt.plot(ma100, 'g', label='100-Day MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Third Plot: With 100 MA and 200 MA
st.subheader('🔵 Closing Price with 100 & 200-Day Moving Averages')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, 'g', label='100-Day MA')
plt.plot(ma200, 'b', label='200-Day MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Split Data
train_df = df['Close'][0:int(len(df) * 0.85)]
test_df = df['Close'][int(len(df) * 0.85):]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_arr = scaler.fit_transform(train_df.values.reshape(-1, 1))

# Load model
try:
    model = load_model('keras_model.h5')
    logger.info("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prepare test data
past_100_days = train_df.tail(100)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.transform(final_df.values.reshape(-1, 1))

x_test, y_test = [], []
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_pred = model.predict(x_test)

# Rescale predictions
scale_factor = 1 / scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Final Plot
st.subheader('📈 Predicted vs Original Prices')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

