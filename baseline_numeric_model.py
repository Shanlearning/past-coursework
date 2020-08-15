    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM with avg price 
@author: effie & Shan
"""
###############################################################################
'load data'

###############################################################################
import numpy as np
from datetime import datetime
from tqdm import tqdm
import math
import re
###############################################################################
import bs4 as bs
import requests

import json
###############################################################################
'to plot within notebook'
import matplotlib.pyplot as plt

###############################################################################
'importing required nn libraries'
import tensorflow as tf
from tensorflow.keras.layers import Dense,  LSTM ,Dropout , Reshape

from sklearn.metrics import mean_squared_error
################################################
import os
os.chdir('C:\\Users\\zhong\\Dropbox\\github\\sp500')

###############################################################################
"""load the name list of s&p 500 companies"""

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    company_names = []
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        ticker = "".join(re.findall("[a-zA-Z]+", ticker))
        tickers.append(ticker)
        company_names.append(row.findAll('td')[1].text)
        
    return tickers , company_names
##################################
    
tickers , company_names = save_sp500_tickers()
###############################################################################
'load adjusted s&p 500 data'
'use adjusted close is better'

data =  json.loads(open('sp500.json').read())

def available_dates():
    available_dates = list(data.keys())
    def myFunc(item):
        return datetime.strptime(item ,'%Y-%m-%d')
    available_dates.sort(reverse=True, key=myFunc)
    return available_dates    

available_dates = available_dates()
available_tickers = ['SPX','BIC','CNY'] + tickers

ticker_names = ['S&P500','Bitcoin','china yuan'] + company_names

dat = []

for ticker in tqdm(available_tickers):
    Lst = []
    nearest = 0
    for date in available_dates:
        if ticker in data[date].keys():
            nearest = math.log( data[date][ticker] )
            Lst.append(nearest)
        else:
            Lst.append(nearest)                
    Lst.reverse()
    dat.append(Lst)

###############################################################################
'process data'
past_window_size=120
future_window_size=1

# interested_y = available_tickers
interested_y = ['SPX'] #,'BIC']

def lstm_data(data , past_window_size , future_window_size ,available_dates, available_tickers , interested_y):
    X= [ ]
    y= [ ]
    
    i=0
    while (i + past_window_size)<=len(available_dates) - future_window_size :    
        X_day = []
        y_day = []
        for _i in range(0,len(available_tickers)):            
            X_day.append(data[_i][i: i + past_window_size ]  )
            if available_tickers[_i] in interested_y :
                y_day.append(data[_i][i + past_window_size : i + past_window_size + future_window_size])
        X.append(X_day)
        y.append(y_day)
        i += 1
    return X,y
        
X , y = lstm_data(dat, past_window_size , future_window_size,available_dates,available_tickers, interested_y)
X = tf.convert_to_tensor(X, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)
y = tf.reshape(y, shape = [len(y), len(interested_y) , future_window_size])

print("X shape is {}".format(X.shape ))
print("y shape is {}".format(y.shape ))

###############################################################################
'train/test test'

def split(X,y,split_rate=0.8):
    'X, y are from window function'
    train_size = int( len(X)*split_rate)
    X_train = X[0:train_size,:]
    y_train = y[0:train_size,:]
    X_test = X[train_size:,:]
    y_test = y[train_size:,:]
    
    return(X_train,y_train,X_test,y_test)

X_train,y_train,X_test,y_test= split(X,y,split_rate=0.8)

print("x_train size is {}".format(X_train.shape ))
print("y_train size is {}".format(y_train.shape ))
print("x_test size is {}".format(X_test.shape ))
print("y_test size is {}".format(y_test.shape ))

###############################################################################
'create and fit the LSTM network'
'method1 , without text input'

def make_model(lstm_units,dropout_rate,past_window_size,future_window_size):
    sequence_input = tf.keras.Input(shape=(len(available_tickers),past_window_size),dtype=tf.float32,name='stock_lstm_input')
    lstm_1 = LSTM(units=lstm_units, return_sequences=True)(sequence_input)
    dropout_1 = Dropout(dropout_rate)(lstm_1)
    lstm_2 = LSTM(units=32, return_sequences=True)(dropout_1)
    dense_1 = Dense(16,kernel_initializer='uniform',activation='relu')(lstm_2)
    dense_2 = Dense(future_window_size * len(interested_y),kernel_initializer='uniform',activation='linear')(dense_1)    
    result = Reshape( (len(interested_y),future_window_size) )(dense_2)    
    model = tf.keras.Model(inputs = sequence_input,
                          outputs = result,
                          name='model1')
    model.compile(loss='mean_squared_error', optimizer='adam')    
    return model

###############################################################################
"record callbacks for epoch and batch size"
class batch_epoch(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_train_loss = []
        self.epoch_test_loss = []
       
    def on_epoch_end(self, epoch, logs= {}):
        self.epoch_train_loss.append( logs['loss'])
        self.epoch_test_loss.append(logs['val_loss'] )
        print('The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss']))
        
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)        
###############################################################################

lstm_units = 120
dropout_rate=0.25

history = json.loads(open('history.json').read())
  
model = make_model(lstm_units=lstm_units,dropout_rate=dropout_rate,past_window_size=120,future_window_size=1)
model_history = batch_epoch()

model.summary()

model.fit(X_train, y_train, epochs=150, batch_size= 256 , validation_data= (X_test,y_test) )

model.save("new.h5")

X_pred = model.predict(X_train)
X_pred_test = model.predict(X_test)

y_train_hat = model.predict(X_train)
y_train_hat_reshape = y_train_hat.reshape(len(y_train_hat),1)
y_train_hat_reshape_true = np.exp(y_train_hat_reshape)

y_train_reshape = np.array(y_train).reshape(len(y_train),1)
y_train_reshape_true = np.exp( y_train_reshape)
RMSE_train = mean_squared_error(y_train_hat_reshape, y_train_reshape)

plt.plot(range(len(y_train)),np.array(y_train).reshape(len(y_train),1),label = 'Truth')
plt.plot(range(len(X_pred)),X_pred.reshape(len(X_pred),1),label = 'Predicted')
plt.ylim([6.2,8])
plt.legend(loc="upper left")
plt.xlabel('Days')
plt.ylabel('S&P 500 index')
plt.title('S&P500 Index 2-stacked LSTM with Texts Predictions Vs. True Values \n  Forward Forcasting for 3980 days -train RMSE={}'.format(RMSE_train))
plt.savefig('train.png', dpi=200)


y_test_hat = model.predict(X_test)
y_test_hat_reshape = y_test_hat.reshape(len(y_test_hat),1)
y_test_hat_reshape_true = np.exp(y_test_hat_reshape)

y_test_reshape = np.array(y_test).reshape(len(y_test),1)
y_test_reshape_true = np.exp( y_test_reshape)
RMSE_test = mean_squared_error(y_test_hat_reshape, y_test_reshape)

plt.plot(range(len(y_test)),np.array(y_test).reshape(len(y_test),1),label = 'Truth')
plt.plot(range(len(X_pred_test)),np.array(X_pred_test).reshape(len(X_pred_test),1),label = 'Predicted')
plt.ylim([7.7,8.4])
plt.legend(loc="upper left")
plt.xlabel('Days')
plt.ylabel('S&P 500 index')
plt.title('S&P500 Index 2-stacked LSTM with Texts Predictions Vs. True Values \n  Forward Forcasting for 3980 days -test RMSE={}'.format(RMSE_test))
plt.savefig('test.png', dpi=200)

model.load_weights("new.h5")


with open('history.json', 'w') as fp:
    json.dump(history, fp)
