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

from tensorflow.keras.layers import Dense,  LSTM ,Dropout , Flatten , Reshape ,Concatenate


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
'load sentiment_data'
padding_length = 5

sentiment_data =  json.loads(open('sentiment_data.json').read())

company_data = {}
for date in tqdm(sentiment_data.keys()):
    company_data[date] = []

for date in tqdm(sentiment_data.keys()):
    for item in sentiment_data[date]:
        if (item['Actor1'] in ['UNITED STATES']) and (item['Actor2'] in ['CHINA']):            
            company_data[date].append(item)
        if (item['Actor1'] in ['CHINA']) and (item['Actor2'] in ['UNITED STATES']):            
            company_data[date].append(item)

Lst = []
for date in company_data.keys():
    for item in company_data[date]:
        Lst.append(item)
len(Lst)

score = []
for item in Lst:
    score.append([item['Sentiment Score']])
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(score)

for item in Lst:
    item['Sentiment Score'] = float(scaler.transform([[item['Sentiment Score']]])[0][0])

sentiment_list = []
action_list = []
for date in available_dates:
    date = datetime.strptime(date ,'%Y-%m-%d')
    date = date.strftime('%Y%m%d')
    if date in company_data.keys():
        temp_sentiment = []
        temp_action = []
        for item in company_data[date]:
            temp_sentiment.append(item['Sentiment Score'])
            zeros = np.zeros(20)
            pos = int(item['Action Code'])-1
            zeros[pos] = 1
            temp_action.append(zeros.tolist())
        sentiment_list.append(temp_sentiment)
        action_list.append(temp_action)
    else:
        sentiment_list.append([])
        action_list.append([])
        
sentiment_list.reverse()
action_list.reverse()

new_sentiment_lst = []
new_action_lst = []

for element in sentiment_list:
    if len(element) > padding_length:
        element = element[0:padding_length]
        new_sentiment_lst.append(element)
    else :
        element = element + [0] *(padding_length-len(element))
        new_sentiment_lst.append(element)

for element in action_list:
    if len(element) > padding_length:
        element = element[0:padding_length]
        new_action_lst.append(element)
    else :
        element = element + [np.zeros(20).tolist()] *(padding_length-len(element))
        new_action_lst.append(element)

###############################################################################

# interested_y = available_tickers
interested_y = ['SPX'] #,['GOOGLE']

'process data'
past_window_size=120
future_window_size=30

###############################################################################

def lstm_data(data , past_window_size , future_window_size ,available_dates, available_tickers, interested_y):
    X= [ ]
    X_sentiment = []
    X_action = []
    y = []
    i=0
    while (i + past_window_size + future_window_size )<=len(available_dates) :    
        X_day = []
        y_day = []
        X_sentiment_day = new_sentiment_lst[i+past_window_size]
        X_action_day = new_action_lst[i+past_window_size]
        for _i in range(0,len(available_tickers)):            
            X_day.append(data[_i][i: i + past_window_size ]  )
            if available_tickers[_i] in interested_y :
                y_day.append(data[_i][i + past_window_size : i + past_window_size + future_window_size])
        X.append(X_day)
        X_sentiment.append(X_sentiment_day)
        X_action.append(X_action_day)
        y.append(y_day)
        i += 1
    return X,X_sentiment,X_action,y

X ,X_sentiment,X_action,y = lstm_data(dat, past_window_size , future_window_size,available_dates,available_tickers, interested_y)

X = tf.convert_to_tensor(X, dtype=tf.float32)

'I made change here'
X_sentiment = tf.convert_to_tensor(new_sentiment_lst, dtype=tf.float32)
X_action = tf.convert_to_tensor(new_action_lst, dtype=tf.float32)

y = tf.convert_to_tensor(y, dtype=tf.float32)
y = tf.reshape(y, shape = [len(y), len(interested_y) , future_window_size])

X_sentiment = X_sentiment[past_window_size:len(X_sentiment)]

X_action = X_action[past_window_size:len(X_action)]

###############################################################################
'train/test test'

def split(X, X_sentiment, X_action, y, split_rate):
    'X, y are from window function'
    train_size = int( len(X)*split_rate)
    X_train = [ X[0:train_size,:] , X_sentiment[0:train_size,:] , X_action[0:train_size,:] ] 
    y_train = y[0:train_size,:]
    X_test = [ X[train_size:,:] , X_sentiment[train_size:,:] , X_action[train_size:,:] ]
    y_test = y[train_size:,:]
    
    return(X_train,y_train,X_test,y_test)

X_train,y_train,X_test,y_test= split(X,X_sentiment,X_action, y,split_rate=0.8)

###############################################################################
'create and fit the LSTM network'
'method1 , without text input'

def make_model(lstm_units,dropout_rate,past_window_size,future_window_size):

    sequence_input = tf.keras.Input(shape=(len(available_tickers),past_window_size),dtype=tf.float32,name='stock_lstm_input') 
    
    lstm_1 = LSTM(units=lstm_units, return_sequences=True)(sequence_input)
    dropout_1 = Dropout(dropout_rate)(lstm_1)
    lstm_2 = LSTM(units=lstm_units, return_sequences=True)(dropout_1)
    dropout_2 = Dropout(dropout_rate)(lstm_2)    
    dense_lstm = Dense(future_window_size)(dropout_2)
    flatten_lstm = Flatten()(dense_lstm)
           
    sentiment_score = tf.keras.Input(shape=(  padding_length),dtype=tf.float32,name='text_sentiment_score')
    sentiment_action = tf.keras.Input(shape=(  padding_length, 20),dtype=tf.float32,name='text_sentiment_action')
    
    dense_sentiment_score = Dense(1)(sentiment_score)
    dense_sentiment_action = Dense(padding_length)(sentiment_action)
    
    flatten_sentiment = Flatten()(dense_sentiment_score)
    flatten_action = Flatten()(dense_sentiment_action)

    concatenate_1 = Concatenate()([flatten_lstm,flatten_sentiment,flatten_action])
    dense_2 = Dense(future_window_size * len(interested_y))(concatenate_1)
    result = Reshape( (len(interested_y),future_window_size) )(dense_2)
    model = tf.keras.Model(inputs = [sequence_input,sentiment_score,sentiment_action] ,
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
        
###############################################################################

lstm_units= 120
dropout_rate=0.25

"define expriment function "

history = json.loads(open('history.json').read())
history.keys() 

model = make_model(lstm_units=lstm_units,dropout_rate=dropout_rate,past_window_size=120,future_window_size=1)
model.summary()

model_history = batch_epoch()
model.fit(X_train, y_train, epochs=150, batch_size= 256 , validation_data= (X_test,y_test) , callbacks= [model_history] )

model.save("text_model.h5")

model.load_weights("text_model.h5")




"plot"
plt.plot(range(len(model_history.epoch_train_loss )),  model_history.epoch_train_loss ,label = 'training' ,color = 'green')
plt.plot(range(len(model_history.epoch_test_loss)), model_history.epoch_test_loss , label = 'validation',color = 'red' )
plt.ylim([0,0.5])
plt.legend(loc="upper right")
plt.xlabel('Number of epochs run')
plt.ylabel('Mean Square Loss')
plt.savefig('text_train_vaild.png', dpi=200)


'vs best'
plt.plot(range(len(model_history.epoch_test_loss )),  model_history.epoch_test_loss ,label = 'with text' ,color = 'green')
plt.plot(range(len(history['256_test_loss_dropout_0.25_lstm_120_stack_2'])),
         history['256_test_loss_dropout_0.25_lstm_120_stack_2'] , label = 'without text',color = 'red' )
plt.ylim([0,0.5])
plt.legend(loc="upper right")
plt.xlabel('Number of epochs run')
plt.ylabel('Mean Square Loss')
plt.savefig('text_vs_numeric.png', dpi=200)

############## plot
X_pred = model.predict(X_train)
X_pred_test = model.predict(X_test)

y_train_hat = model.predict(X_train)
y_train_hat_reshape = y_train_hat.reshape(len(y_train_hat),1)
y_train_hat_reshape_true = np.exp(y_train_hat_reshape)

y_train_reshape = np.array(y_train).reshape(len(y_train),1)
y_train_reshape_true = np.exp( y_train_reshape)
RMSE_train = mean_squared_error(y_train_hat_reshape, y_train_reshape)

plt.plot(range(len(y_train)),np.array(y_train).reshape(len(y_train),1),label = 'Truth')
plt.plot(range(len(X_pred)),X_pred.reshape(len(X_pred),1),label = 'Prediction')
plt.ylim([6.2,8])
plt.legend(loc="upper left")
plt.xlabel('Days')
plt.ylabel('S&P 500 index')
plt.title('S&P500 Index 2-stacked LSTM with Texts Predictions Vs. True Values \n  Forward Forcasting for 3980 days -train RMSE={}'.format(RMSE_train))
plt.savefig('train.png', dpi=200)

'second plot'

y_test_hat = model.predict(X_test)
y_test_hat_reshape = y_test_hat.reshape(len(y_test_hat),1)
y_test_hat_reshape_true = np.exp(y_test_hat_reshape)

y_test_reshape = np.array(y_test).reshape(len(y_test),1)
y_test_reshape_true = np.exp( y_test_reshape)
RMSE_test = mean_squared_error(y_test_hat_reshape, y_test_reshape)

plt.plot(range(len(y_test)),np.array(y_test).reshape(len(y_test),1),label = 'Truth')
plt.plot(range(len(X_pred_test)),np.array(X_pred_test).reshape(len(X_pred_test),1),label = 'Prediction')
plt.ylim([7.5,8.2])
plt.legend(loc="upper left")
plt.xlabel('Days')
plt.ylabel('S&P 500 index')
plt.title('S&P500 Index 2-stacked LSTM with Texts Predictions Vs. True Values \n  Forward Forcasting for 3980 days -test RMSE={}'.format(RMSE_test))
plt.savefig('test.png', dpi=200)


#history['256_train_loss_dropout_0.25_lstm_120_stack_3'] = model_history.epoch_train_loss
#history['256_test_loss_dropout_0.25_lstm_120_stack_3'] = model_history.epoch_test_loss

with open('history.json', 'w') as fp:
    json.dump(history, fp)
