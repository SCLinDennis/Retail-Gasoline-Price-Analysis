#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:49:33 2019

@author: DennisLin
"""
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt 


train = pd.read_pickle('./train.pkl')
prediction = pd.read_pickle('./prediction.pkl')
plt.figure()
plt.plot(train)
plt.plot(prediction['Prediction'])
plt.title('Gas Price Forecasting')
plt.xlabel('Year')
plt.ylabel("Gas Price per gallon ($/gal)")
plt.legend()
plt.show()
plt.savefig('./forecasting60.png')