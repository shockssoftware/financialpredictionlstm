from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
import plotly.graph_objects as go
# libraries
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home.html')
def home():
    return render_template('home.html')


@app.route('/Contact.html')
def contact():
    return render_template('Contact.html')

@app.route('/Predictor.html')
def Predict():
    return render_template('Predictor.html')

@app.route('/stocks/<string:name>',methods=['GET'])
def data(name):
  stock = name
  # get data / handle time / convert to numpy array
# get data / handle time / convert to numpy array
  now = int(time.time()) 
  now =str(now)
  days_delta = int(time.time()) - 12500000
  days =str(days_delta)
  r = requests.get('https://finnhub.io/api/v1/stock/candle?symbol='+stock+'&from='+days+'&to='+now+'&resolution=D&token=c36j4jqad3ifoi8hsu50')
  j = r.json() 
  df = pd.DataFrame.from_dict(j)
  df.t = (pd.to_datetime(df['t'],unit='s'))
  df_close = df[['t','c']]
  all_close = df_close['c'].values.astype(float)
 
# model building
  train=all_close[:-15].reshape(-1, 1)
  test=all_close[-15:].reshape(-1, 1)
  scaler=MinMaxScaler()
  scaled_train=scaler.fit_transform(train)
  scaled_test=scaler.transform(test)
 
  n_input= len(train) - 1
  n_features=1
 
  train_generator=TimeseriesGenerator(scaled_train,
                                     scaled_train,
                                      n_input,
                                      batch_size=1)

  early_stopping = EarlyStopping()
 
  custom_early_stopping = EarlyStopping(   
    monitor='loss', 
    patience=35, 
    min_delta=0.009, 
    mode='min' # was max
  )
  model=Sequential()
  model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
  model.add(LSTM(50,activation='relu',return_sequences=True))
  model.add(LSTM(10,activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam',loss='mse')
 
##
 
  model.fit(train_generator,epochs=140, callbacks=[custom_early_stopping])
 
  test_predictions = []
#Select last n_input values from the train data
  first_eval_batch = scaled_train[-n_input:]
#reshape the data into LSTM required (#batch,#timesteps,#features)
  current_batch = first_eval_batch.reshape((1, n_input, n_features))
  for i in range(len(test)):
# get prediction, grab the exact number using the [0]
    pred = model.predict(current_batch)[0]
# Add this prediction to the list
    test_predictions.append(pred)
# The most critical part, update the (#batch,#timesteps,#features
# using np.append(
# current_batch[:        ,1:   ,:] ---------> read this as
# current_batch[no_change,1:end,no_change]
# (Do note the second part has the timesteps)
# [[pred]] need the double brackets as current_batch is a 3D array
# axis=1, remember we need to add to the second part i.e. 1st axis
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
 
 
  actual_predictions = scaler.inverse_transform(test_predictions)
  rows =list(range(len(train), len(train)+15)) # this
  pred_df = pd.DataFrame(data=actual_predictions,index=rows)
 
  future_train = all_close.reshape(-1,1)
  scaled_future_train=scaler.fit_transform(future_train)
  n_input= len(future_train) - 1
  n_features=1
 
  future_generator=TimeseriesGenerator(scaled_future_train,
                                     scaled_future_train,
                                      n_input,
                                      batch_size=1)
  custom_early_stopping = EarlyStopping(   
    monitor='loss', 
    patience=35, 
    min_delta=0.009, 
    mode='min'
  )
 
  model=Sequential()
  model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
  model.add(LSTM(50,activation='relu',return_sequences=True))
  model.add(LSTM(10,activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam',loss='mse')
 
 
 
 
  model.fit(future_generator,epochs=140, callbacks=[custom_early_stopping])
 
  future_predictions = []
#Select last n_input values from the train data
  first_eval_batch = scaled_future_train[-n_input:]
#reshape the data into LSTM required (#batch,#timesteps,#features)
  current_batch = first_eval_batch.reshape((1, n_input, n_features))
  for i in range(len(test)):
# get prediction, grab the exact number using the [0]
    pred = model.predict(current_batch)[0]
# Add this prediction to the list
    future_predictions.append(pred)
# The most critical part, update the (#batch,#timesteps,#features
# using np.append(
# current_batch[:        ,1:   ,:] ---------> read this as
# current_batch[no_change,1:end,no_change]
# (Do note the second part has the timesteps)
# [[pred]] need the double brackets as current_batch is a 3D array
# axis=1, remember we need to add to the second part i.e. 1st axis
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
 
 
  future_actual_predictions = scaler.inverse_transform(future_predictions)
  rows =list(range(len(future_train),len(future_train)+15))
  future_pred_df = pd.DataFrame(data=future_actual_predictions,index=rows)
  fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'])])
  fig.update_layout(xaxis_rangeslider_visible=True)
  fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df[0], mode="lines",name='Test_Prediction'))
  fig.add_trace(go.Scatter(x=future_pred_df.index, y=future_pred_df[0], mode="lines",name='Future_Prediction'))
  fig.update_layout(title=stock+' Stock Price candlestick data, test prediction & future Prediction at market close')
  fig.update_layout(template='plotly_dark')
  fig.update_yaxes(title='$')
  fig.update_xaxes(title='Days')
  fig.write_html('templates/'+stock+' Pred.html')
  return render_template(stock+' Pred.html')

@app.route('/daytrade/<string:name>',methods=['GET'])
def get_data(name):
  stock = name
  # get data / handle time / convert to numpy array
# get data / handle time / convert to numpy array
  now = int(time.time()) 
  now =str(now)
  days_delta = int(time.time()) - 125000 * 2
  days =str(days_delta)
  r = requests.get('https://finnhub.io/api/v1/stock/candle?symbol='+stock+'&from='+days+'&to='+now+'&resolution=5&token=c36j4jqad3ifoi8hsu50')
  j = r.json() 
  df = pd.DataFrame.from_dict(j)
  df.t = (pd.to_datetime(df['t'],unit='s'))
  df_close = df[['t','c']]
  all_close = df_close['c'].values.astype(float)
 
# model building
  train=all_close[:-15].reshape(-1, 1)
  test=all_close[-15:].reshape(-1, 1)
  scaler=MinMaxScaler()
  scaled_train=scaler.fit_transform(train)
  scaled_test=scaler.transform(test)
 
  n_input= len(train) - 1
  n_features=1
 
  train_generator=TimeseriesGenerator(scaled_train,
                                     scaled_train,
                                      n_input,
                                      batch_size=1)

  early_stopping = EarlyStopping()
 
  custom_early_stopping = EarlyStopping(   
    monitor='loss', 
    patience=35, 
    min_delta=0.009, 
    mode='min' # was max
  )
  model=Sequential()
  model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
  model.add(LSTM(50,activation='relu',return_sequences=True))
  model.add(LSTM(10,activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam',loss='mse')
 
##
 
  model.fit(train_generator,epochs=140, callbacks=[custom_early_stopping])
 
  test_predictions = []
#Select last n_input values from the train data
  first_eval_batch = scaled_train[-n_input:]
#reshape the data into LSTM required (#batch,#timesteps,#features)
  current_batch = first_eval_batch.reshape((1, n_input, n_features))
  for i in range(len(test)):
# get prediction, grab the exact number using the [0]
    pred = model.predict(current_batch)[0]
# Add this prediction to the list
    test_predictions.append(pred)
# The most critical part, update the (#batch,#timesteps,#features
# using np.append(
# current_batch[:        ,1:   ,:] ---------> read this as
# current_batch[no_change,1:end,no_change]
# (Do note the second part has the timesteps)
# [[pred]] need the double brackets as current_batch is a 3D array
# axis=1, remember we need to add to the second part i.e. 1st axis
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
 
 
  actual_predictions = scaler.inverse_transform(test_predictions)
  rows =list(range(len(train), len(train)+15)) # this
  pred_df = pd.DataFrame(data=actual_predictions,index=rows)
 
  future_train = all_close.reshape(-1,1)
  scaled_future_train=scaler.fit_transform(future_train)
  n_input= len(future_train) - 1
  n_features=1
 
  future_generator=TimeseriesGenerator(scaled_future_train,
                                     scaled_future_train,
                                      n_input,
                                      batch_size=1)
  custom_early_stopping = EarlyStopping(   
    monitor='loss', 
    patience=35, 
    min_delta=0.009, 
    mode='min'
  )
 
  model=Sequential()
  model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
  model.add(LSTM(50,activation='relu',return_sequences=True))
  model.add(LSTM(10,activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam',loss='mse')
 
 
 
 
  model.fit(future_generator,epochs=140, callbacks=[custom_early_stopping])
 
  future_predictions = []
#Select last n_input values from the train data
  first_eval_batch = scaled_future_train[-n_input:]
#reshape the data into LSTM required (#batch,#timesteps,#features)
  current_batch = first_eval_batch.reshape((1, n_input, n_features))
  for i in range(len(test)):
# get prediction, grab the exact number using the [0]
    pred = model.predict(current_batch)[0]
# Add this prediction to the list
    future_predictions.append(pred)
# The most critical part, update the (#batch,#timesteps,#features
# using np.append(
# current_batch[:        ,1:   ,:] ---------> read this as
# current_batch[no_change,1:end,no_change]
# (Do note the second part has the timesteps)
# [[pred]] need the double brackets as current_batch is a 3D array
# axis=1, remember we need to add to the second part i.e. 1st axis
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
 
 
  future_actual_predictions = scaler.inverse_transform(future_predictions)
  rows =list(range(len(future_train),len(future_train)+15))
  future_pred_df = pd.DataFrame(data=future_actual_predictions,index=rows)
  fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'])])
  fig.update_layout(xaxis_rangeslider_visible=True)
  fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df[0], mode="lines",name='Test_Prediction'))
  fig.add_trace(go.Scatter(x=future_pred_df.index, y=future_pred_df[0], mode="lines",name='Future_Prediction'))
  fig.update_layout(title=stock+' Stock Price 5 minute candlestick data, test prediction & future Prediction for next 15 candles')
  fig.update_layout(template='plotly_dark')
  fig.update_yaxes(title='$')
  fig.update_xaxes(title='Days')
  fig.write_html('templates/'+stock+' Pred.html')
  return render_template(stock+' Pred.html')

if __name__ == "__main__":
  app.run(debug=True)
