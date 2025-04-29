
import pandas as pd 
import numpy as np
# from sqlalchemy import label
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt 
from keras.models import load_model 

model = load_model(r'C:\Users\geniu\finalproject1\StocksModel.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol' , 'TSLA')

start_date = '2010-02-24'
end_date = '2025-04-24'
# stock_symbol = 'TSLA'

stock_data = yf.download(stock , start_date , end_date)

st.subheader('Stock Data')
st.write(stock_data)

stock_data_train = stock_data.Close[0: int(len(stock_data)*0.80)]
stock_data_test = stock_data.Close[int(len(stock_data)*0.80) : len(stock_data)]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = stock_data_train.tail(100)

stock_data_test = pd.concat([past_100_days , stock_data_test] , ignore_index = True)

stock_data_test_scale = scaler.fit_transform(stock_data_test)

st.subheader('Price vs MA_50')
MA_50_days = stock_data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(MA_50_days , 'r' , label = 'MA_50' )
plt.plot(stock_data.Close , 'b' , label = 'Closing Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA_50 vs MA_100')
MA_100_days = stock_data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(MA_50_days , 'r' , label = 'MA_50')
plt.plot(stock_data.Close , 'b' , label = 'Closing Price')
plt.plot(MA_100_days , 'g' , label = 'MA_100')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA_100 vs MA_200')
MA_200_days = stock_data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(MA_100_days , 'r' , label = 'MA_100' )
plt.plot(stock_data.Close , 'b' , label = 'Closing Price')
plt.plot(MA_200_days , 'g' , label='MA_200')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig3)

#Now we again perform array slicing
x = []
y = []

for i in range(100, stock_data_test_scale.shape[0]):
    x.append(stock_data_test_scale[i-100:i])
    y.append(stock_data_test_scale[i,0])
#Taking the first 100 days data to predict the next one at one data 

x , y = np.array(x) , np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict*scale
y = y*scale

st.subheader('Original Price vs Predicted Price ')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict , 'b' , label = 'Predicted Price')
plt.plot(y , 'r' , label = 'Original Price' )
plt.legend()
plt.xlabel('')
plt.ylabel('')
plt.show()
st.pyplot(fig4)
