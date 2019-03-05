#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:56:28 2019

@author: DennisLin
"""

import numpy as np
from pandas import Series
import pandas as pd
#from statsmodels.tsa.arima_model import ARIMA
from pyquery import PyQuery as pq
#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt 


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def preprocess(string):
    '''
    Change the type from string to list and remove redundant ''
    Args:
        string: (str) string loaded from pyquery
    Return:
        data: (list) 
    '''
    string = string.replace(',', '').replace('.', '').replace('NA', '0').split(' ')
    
    return list(filter(lambda a: a != '', string))

def url2df(url, name):
    '''
    use pyquery to crawl the data and convert it to pandas.dataframe
    Args:
        url: (str) website url for crawling
        name: (str) the name of the data
    Return:
        data: (DataFrame) 
    '''
    html_doc = pq(url)
    html_doc.contents()
    
    value = ".B3:nth-child(3)"
    date = ".B6"
    
    value_data = preprocess(html_doc(value).text())
    value_data = list(map(int, value_data))
    date_data = preprocess(html_doc(date).text())    
    data = pd.DataFrame(value_data, columns = [name], index = date_data)
    return data

def dict2df(dic):
    '''
    Transform a dictionary object to dataframe
    Args:
        dic: (dictionary)
    Return: 
        df: (DataFrame)
    '''
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df(url, name)], axis = 1)
    return df

price_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPM0_PTE_R10_DPG&f=W'
Region = ["East Coast", "Midwest", "Gulf Coast", "Rocky Mountain", "West Coast"]
train_sample = 10
if __name__ == "__main__":
    price_dict = {}
    
    for i, r in enumerate(Region):
        price_dict[r] = price_url.replace('10', '%d') %(10*(i+1))

    df_price = dict2df(price_dict).div(1000)
    df_price.index = pd.to_datetime(df_price.index)

#%%

#creating train and test sets
dataset = df_price[Region[0]].values
new_data = df_price[Region[0]]
dataset = dataset[:, None]
train = dataset[0:300,:]
valid = dataset[300:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(train_sample,len(train)):
    x_train.append(scaled_data[i-train_sample:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(df_price) - len(valid) - train_sample:len(df_price) - len(valid)].values
#inputs = new_data[len(df_price) - len(valid) - 20:].values

inputs = inputs.reshape(-1,1)
inputs_list = []
for i in range(train_sample):
    inputs_list.append(float(inputs[i]))
#inputs_list = inputs.tolist()
tmp  = scaler.transform(inputs)
inputs_list2 = []
for i in range(train_sample):
    inputs_list2.append(float(tmp[i]))
#inputs_list2 = list(inputs_list2)

#X_test = []
#for i in range(20,inputs.shape[0]+1):
#    X_test.append(inputs[i-20:i,0])
#X_test = np.array(X_test)
#
#X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
test_temp = []
test = []
for i in range(36):
#    tmp = np.expand_dims(X_test[i, :, 0], axis=0)
#    tmp = X_test[0, :, 0]
    tmp = inputs_list2
    temp_array = np.zeros(([1,train_sample, 1]))
    temp_array[0, :, 0] = np.asarray(tmp).reshape([1, train_sample])
    print(temp_array)
#    temp_result = scaler.inverse_transform(model.predict(temp_array))
    test_temp.append(scaler.inverse_transform(model.predict(temp_array)))
#    print()
    test.append(test_temp[-1][0][0])
    inputs_list.pop(0)
    inputs_list.append(float(test_temp[-1]))
    print(i)
    inputs_list0 = np.asarray(inputs_list).reshape(-1,1)
    inputs_list2  = scaler.transform(inputs_list0)
    inputs_list2 = list(inputs_list2)

closing_price = test
##closing_price = model.predict(X_test)
#closing_price = 

train = new_data[:300]
#valid = new_data[300:]
#valid = valid.to_frame()
prediction = pd.DataFrame(closing_price, columns= ['prediction'] , index =pd.date_range(start='2018/4/1', periods=36, freq='MS'))
#valid['Predictions'] = closing_price
plt.figure()
plt.plot(train)
plt.plot(prediction)
plt.title('Forecasting of the Gas Price')
plt.xlabel('Year')
plt.ylabel("Gas Price per gallon ($/gal)")
#plt.savefig('./forecasting40.png')

#
#
#
#plt.figure()
#plt.plot(range(closing_price.shape[0]), closing_price)
#plt.show()