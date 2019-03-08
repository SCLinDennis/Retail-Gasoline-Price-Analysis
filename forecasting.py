#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:56:28 2019

@author: DennisLin
"""

import numpy as np
import pandas as pd
from pyquery import PyQuery as pq
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt 


def preprocess(string):
    '''
    Change the type from string to list and remove redundant ''
    Args:
        string: (str) string loaded from pyquery
    Return:
        data: (list) 
    '''
    assert isinstance(string, str)
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
    assert isinstance(url, str)
    assert isinstance(name, str)
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
    assert isinstance(dic, dict)
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df(url, name)], axis = 1)
    return df

class Model():
    def __init__(self, input_size):#(x_train.shape[1],1)
        '''
        Init the input shape, and build and compile it. 
        '''
        self.input_size = input_size
        self.build_model()
        self.compile_()
    
    def build_model(self):
        '''
        Build model using 2 layers of LSTM and one hidden layer
        '''
        model = Sequential()
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.input_size)))
        model.add(LSTM(units = 50))
        model.add(Dense(1))
        self.model = model
        
    def compile_(self):
        '''
        plug in the model with mean square error loss and adma optimizer.
        '''
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
    def fit(self, train, label):
        '''
        fit the model with training data and training label
        '''
        self.model.fit(train, label, epochs=1, batch_size=1, verbose=2)
        
    def predict(self, test):
        '''
        predict the given test data
        '''
        return self.model.predict(test)
        
        
        

price_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPM0_PTE_R10_DPG&f=W'
Region = ["East Coast", "Midwest", "Gulf Coast", "Rocky Mountain", "West Coast"]
train_sample = 50
n_predict = 60
if __name__ == "__main__":
    price_dict = {}
    
    for i, r in enumerate(Region):
        price_dict[r] = price_url.replace('10', '%d') %(10*(i+1))

    df_price = dict2df(price_dict).div(1000)
    df_price.index = pd.to_datetime(df_price.index)

    
    #creating train and test sets
    df_price['Gas Price Trend'] = df_price.mean(axis=1)
    dataset = df_price['Gas Price Trend'].values
    new_data = df_price['Gas Price Trend']
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
    model = Model((x_train.shape[1],1))
    model.fit(x_train, y_train)
    
    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(df_price) - len(valid) - train_sample:len(df_price) - len(valid)].values
    inputs = inputs.reshape(-1,1)
    inputs_list = []
    for i in range(train_sample):
        inputs_list.append(float(inputs[i]))
    
    tmp  = scaler.transform(inputs)
    inputs_list2 = []
    for i in range(train_sample):
        inputs_list2.append(float(tmp[i]))
    
    test_temp = []
    test = []
    for i in range(n_predict):
        tmp = inputs_list2
        temp_array = np.zeros(([1,train_sample, 1]))
        temp_array[0, :, 0] = np.asarray(tmp).reshape([1, train_sample])
        test_temp.append(scaler.inverse_transform(model.predict(temp_array)))
        test.append(test_temp[-1][0][0])
        inputs_list.pop(0)
        inputs_list.append(float(test_temp[-1]))
        inputs_list0 = np.asarray(inputs_list).reshape(-1,1)
        inputs_list2  = scaler.transform(inputs_list0)
        inputs_list2 = list(inputs_list2)
    
    closing_price = test
    train = new_data[:300]
    test = new_data[300:]
    prediction = pd.DataFrame(closing_price, columns= ['Prediction'] , index =pd.date_range(start='2018/4/1', periods=n_predict, freq='MS'))
    
    plt.figure()
    plt.plot(new_data)
    plt.plot(prediction['Prediction'])
    plt.title('Gas Price Forecasting')
    plt.xlabel('Year')
    plt.ylabel("Gas Price per gallon ($/gal)")
    plt.grid(axis = 'y')
    plt.legend()
    plt.show()
#    plt.savefig('./forecasting60.png')
