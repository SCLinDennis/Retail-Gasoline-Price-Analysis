#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:34:00 2019

@author: DennisLin
"""
#https://docs.google.com/document/d/14yTbyoFJks_aDMtJLcwqm7MG5n-DEk_aJzrVTy8EJUA/edit
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   
import pandas as pd
import matplotlib.pyplot as plt 
from pyquery import PyQuery as pq


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
    
    value = ".B3"
    date = ".B5"
    
    value_data = preprocess(html_doc(value).text())
    value_data = list(map(int, value_data))
    date_data = preprocess(html_doc(date).text())    
    data = pd.DataFrame(value_data, columns = [name], index = date_data)
    return data

def url2np(url, name):
    '''
    use pyquery to crawl the data and convert it to numpy array
    Args:
        url: (str) website url for crawling
        name: (str) the name of the data
    Return:
        data: (numpy array) 
    '''
    html_doc = pq(url)
    html_doc.contents()
    
    value = ".B3"
    date = ".B5"
    
    value_data = preprocess(html_doc(value).text())
    value_data = list(map(int, value_data))
    date_data = preprocess(html_doc(date).text())    
#    value_array = np.array(value_data)
    array = np.array([date_data, value_data])
#    return np.hstack((date_array, value_array))
    return array

def dict2df(dic):
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df(url, name)], axis = 1)
    return df

def get_correlate(df_price, df_factor):
    coor = {}
    for i, r in enumerate(Region):
        price_array = normalized(df_price[[r]].values)
        factor_array = normalized(df_factor[[r]].values)        
        min_length = min([price_array.shape[0], factor_array.shape[0]])
        price_array, factor_array = price_array[-min_length:], factor_array[-min_length:]
        coor[r] = np.correlate(price_array.squeeze() , factor_array.squeeze())/min_length
    return coor

def normalized(numpydata):
    max_ = np.max(numpydata)
    min_ = np.min(numpydata)
    return (numpydata-min_)/(max_-min_)
#%%
Region = ['East', 'Midwest', 'GC', 'RM', 'WC']
price_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPM0_PTE_R10_DPG&f=W'
import_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WTTIM_R10-Z00_2&f=W'
stock_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTP11&f=W'
#export_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTEXP11&f=M'
refine_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCRRIP12&f=W'#crude oil input
if __name__ == "__main__":
    gas_price = {}
    price_dict, import_dict, stock_dict, export_dict, refine_dict  = {}, {}, {}, {}, {}

    #store url
    for i, r in enumerate(Region):
        price_dict[r] = price_url.replace('10', '%d') %(10*(i+1))
        import_dict[r] = import_url.replace('10', '%d') %(10*(i+1))
        stock_dict[r] = stock_url.replace('11', '%d') %(10*(i+1) + 1)
#        export_dict[r] = export_url.replace('11', '%d') %(10*(i+1) + 1)
        refine_dict[r] = refine_url.replace('12', '%d') %(10*(i+1) + 2)

    #load url
    df_price = dict2df(price_dict)
    df_import = dict2df(import_dict)
    df_stock = dict2df(stock_dict)
#    df_export = dict2df(export_dict)
    df_refine = dict2df(refine_dict)
    

    #plot figure
    price_fig = df_price.plot.line().get_figure()#1993~Apr, 2019~Feb
    import_fig = df_import.plot.line().get_figure()#2004~Apr, 2019~Feb
    stock_fig = df_stock.plot.line().get_figure()#1990~Jan, 2019~Feb.
#    export_fig = df_export.plot.line().get_figure()
    refine_fig = df_refine.plot.line().get_figure()
    
    
    #save figure    
    price_fig.savefig('price.png')
    import_fig.savefig('import.png')
    stock_fig.savefig('stock.png')
#    export_fig.savefig('export.png')
    refine_fig.savefig('refine.png')    
    
#    price_array = df_price.values
#    plt.figure()
#    plt.scatter(range(1350), price_array[:, 0])
#    import_array = df_import.values
#    stock_array = df_stock.values
    import_cor = get_correlate(df_price, df_import)
    stock_cor = get_correlate(df_price, df_stock)
    refine_cor = get_correlate(df_price, df_refine)