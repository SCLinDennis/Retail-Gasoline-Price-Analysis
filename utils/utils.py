All_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
                   'Connecticut', 'Delaware', 'Dist. of Col.', 
                   'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
                   'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 
                   'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
                   'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
                   'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 
                   'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
                   'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
                   'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin',
                   'Wyoming']
WC = ('Washington', 'Oregon', 'California', 'Nevada', 'Arizona', 'Alaska', 'Hawaii')
RM = ('Montana', 'Idaho', 'Wyoming', 'Utah', 'Colorado')
GC = ('New Mexico',  'Texas', 'Arkansas', 'Louisiana',  'Mississippi', 'Alabama')
Midwest = ('North Dakota',  'South Dakota', 'Nebraska',  'Kansas', 'Oklahoma', 
           'Minnesota', 'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 'Indiana', 'Kentucky',
           'Michigan',  'Tennessee', 'Ohio')

East = ('Florida',  'Georgia', 'South Carolina', 'North Carolina', 'Virginia', 'West Virginia', 
              'Maryland',  'Delaware',  'Pennsylvania',  'New Jersey',  'New York',  'Connecticut', 
              'Rhode Island',  'Vermont',  'New Hampshire',  'Massachusetts',  'Maine', 'Dist. of Col.')
state_regions = [East, Midwest, GC, RM, WC]
Region = ["East Coast", "Midwest", "Gulf Coast", "Rocky Mountain", "West Coast"]
price_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPM0_PTE_R10_DPG&f=W'
import_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WTTIM_R10-Z00_2&f=W'
stock_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTP11&f=W'
population_path = './population/nst-est2018-01.xlsx'

import xlrd
import matplotlib
matplotlib.use('TkAgg')   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import *
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from pyquery import PyQuery as pq

def downsample_other(df, start, end):
    return df.iloc[(df.index.year >= int(start)) & (df.index.year <= int(end))]

def get_income(path):
    df = pd.read_excel(path)
    income = defaultdict(list)
    for i in range(5,57):
        for j in range(1,75,2):
            if j == 5 or j == 15:
                continue
            income[df.iloc[i][0]].append(df.iloc[i][j])
    d = pd.DataFrame(income)
    d = d.iloc[::-1, :]
    tmp = dict()
    for i in range(0,len(d)):
        tmp[i] = str(2018-i)
    d = d.rename(index = tmp)
    d = d.rename(columns = {'D.C.':'Dist. of Col.'})
    income_list = []
    income_dict = defaultdict(lambda: defaultdict(list))
    for num,k in enumerate(state_regions):
        for j in range(0,len(d)):
            for i in k:
                income_dict[Region[num]][1984+j].append(d.iloc[j][i])
    for reg,data_ in income_dict.items():
        tmp = []
        for year,income_ in data_.items():
            income_dict[reg][year] = sum(income_)/len(income_)
    return pd.DataFrame(income_dict)

def process_GDP(data):
    data.at[9, 'GeoName'] = 'Dist. of Col.'
    data = data[data['GeoName'].isin(All_states)]
    data = data.T
    data = data.rename(columns=data.iloc[0])
    data = data.drop(data.index[0])
    for i, region in enumerate(state_regions):
        data[Region[i]] = data[list(region)].sum(axis=1)
    return data.iloc[:, -5:]
    
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

def url2df_im(url, name):
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
    value_data = [value_data[0]] + value_data
    value_data =  list(map(int, value_data))
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
    assert isinstance(url, str)
    assert isinstance(name, str)
    html_doc = pq(url)
    html_doc.contents()
    
    value = ".B3"
    date = ".B5"
    
    value_data = preprocess(html_doc(value).text())
    value_data = list(map(int, value_data))
    date_data = preprocess(html_doc(date).text())    
    array = np.array([date_data, value_data])
    return array

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

def dict2df_im(dic):
    '''
    Transform a dictionary object to dataframe, while dealing with the edge case on `import`
    Args: 
        dic: (dictionary)
    Return: 
        df: (DataFrame)
    '''
    assert isinstance(dic, dict)
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df_im(url, name)], axis = 1)
    return df

def get_correlate(df_price, df_factor):
    '''
    Retuen the correlation of two df.dataframe
    Args:
        df_price: (pd.DataFrame)
        df_factor: (pd.DataFrame)
    Return:
        coor: (dict)
    '''
    assert isinstance(df_price, pd.DataFrame)
    assert isinstance(df_factor, pd.DataFrame)
    coor = {}
    for i, r in enumerate(Region):
        price_array = normalized(df_price[[r]].values)
        factor_array = normalized(df_factor[[r]].values)        
        min_length = min([price_array.shape[0], factor_array.shape[0]])
        price_array, factor_array = price_array[-min_length:], factor_array[-min_length:]
        coor[r] = np.correlate(price_array.squeeze() , factor_array.squeeze())/min_length
    return coor

def normalized(numpydata):
    '''
    Normalized the numpy array.
    Args:
        numpydata: (nparray)
    '''
    assert isinstance(numpydata, np.ndarray)
    max_ = np.max(numpydata)
    min_ = np.min(numpydata)
    return (numpydata-min_)/(max_-min_)

def name_process(array):
    '''
    preprocess the naming conflict between two object
    '''
    assert isinstance(array, np.ndarray)
    out = []
    for i, region in enumerate(state_regions):
        out.append(array.item().get(region))
    return out

def create_corr_map(corr_dict):
    '''
    create the dictionary suitable for mapping to the US map
    Args:
        corr_dict: (dictionary) 
    Return:
        corr_map: (dictionary)
    '''
    assert isinstance(corr_dict, dict)
    corr_map = {}
    for i, region in enumerate(state_regions):
        for state in region:
            corr_map[state] = corr_dict[Region[i]][0]
    return corr_map

def downsample(df, start_year, end_year):
    '''
    Downsample the pandas.dataframe to the data in some period of years.
    Args:
        df: (pandas.dataframe)
        start_year: (int)
        end_year: (int)
    Return:
        df_price_down: (pandas.dataframe)
    '''
    assert isinstance(df, pd.DataFrame)
    assert start_year >= df.index.year[0]
    assert end_year <= df.index.year[-1]
    # df.index = pd.to_datetime(df.index)
    df_tmp = df.groupby(df.index.year).transform('mean')
    df_price_down = df_tmp.iloc[(df_tmp.index.month == 2) & (df_tmp.index.year >= start_year) & (df_tmp.index.year <= end_year)]
    return df_price_down

def corr_bar(import_cor,vehicle_cor,pop_cor,gdp_cor,income_cor, figsize):
    '''
    plot the bar chart of correlation
    Args:
        import_cor:(dictionary)
        pop_cor:(dictionary)
        vehicle:(dictionary)
        gdp_cor:(dictionary)
        income_cor:(dictionary)
        figsize: (tuple)
    '''
    assert isinstance(import_cor, dict)
    assert isinstance(pop_cor, dict)
    assert isinstance(vehicle_cor, dict)
    assert isinstance(gdp_cor, dict)
    assert isinstance(income_cor, dict)
    assert isinstance(figsize, tuple)
    imp_value, vhc_value, pop_value, gdp_value, income_value = [], [], [], [], []
    region_name,region_name2,region_name3,region_name4,region_name5=[], [], [], [], []

    # for reg, val in import_cor.items():
    for reg in Region:
        region_name.append(reg)
        imp_value.append(round(import_cor[reg][0],3))
        region_name2.append(reg)
        vhc_value.append(round(vehicle_cor[reg][0],3))
        region_name3.append(reg)
        pop_value.append(round(pop_cor[reg][0],3))
        region_name4.append(reg)
        gdp_value.append(round(gdp_cor[reg][0],3))
        region_name5.append(reg)
        income_value.append(round(income_cor[reg][0],3))
    assert(region_name==region_name2)
    assert(region_name==region_name3)
    imp=tuple(imp_value)
    vhc=tuple(vhc_value)
    pop=tuple(pop_value)
    gdp=tuple(gdp_value)
    income=tuple(income_value)
    region=tuple(region_name)
    
    ind = np.arange(len(imp))  # the x locations for the groups
    width = 0.1  # the width of the bars
    
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(ind - 2*width, imp, width, 
                    color='SkyBlue', label='Imports vs. Gas Price')
    rects2 = ax.bar(ind - width , vhc, width,
                    color='IndianRed', label='Vehicles vs. Gas Price')
    rects3 = ax.bar(ind, pop, width,
                    color='Purple', label='Population vs. Gas Price')
    rects4 = ax.bar(ind + 1*width, gdp, width, label='GDP vs. Gas Price')
    rects5 = ax.bar(ind + 2*width, income, width, label='Income vs. Gas Price')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Raw correlation')
    ax.set_title('Comparison of Correlations')
    ax.set_xticks(ind)
    ax.set_xticklabels(region)
    ax.grid(axis = 'y')
    ax.legend()
    
    
    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.
    
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
    
        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43} 
    
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
    
    
    autolabel(rects1, "center")
    autolabel(rects2, "center")
    autolabel(rects3, "center")
    autolabel(rects4, "center")
    autolabel(rects5, "center")
    plt.show()

def load_vehicle(num_year):  
    '''
    load the preprocessed vehicle data
    '''      
    year = []
    for year_i in range(num_year):
        year.append(str(1997+year_i))
        npy_file = "./vehicle/dic_{}.npy".format(year_i + 1997)
        tmp_array = name_process(np.load(npy_file, allow_pickle = True))
        new_vehicle = np.array(tmp_array).reshape([1, -1]) 
        if year_i == 0:
            vehicles = new_vehicle
        else:
            vehicles = np.vstack((vehicles, new_vehicle))
    return pd.DataFrame(vehicles, columns = Region, index = year).div(10**8)

def load_population(path):
    '''
    load the population data
    '''
    assert isinstance(path, str)
    wb = xlrd.open_workbook(path) 
    sheet = wb.sheet_by_index(0) 
    population = np.zeros((5, 9))    
    left = 9
    right = 60
    coloumn_total = 15
    year = []
    for iter_year in range(9):
        year.append(str(2010+iter_year))
        for iter_table in range(left, right):
            for iter_region in range(5):
                if All_states[iter_table-left] in state_regions[iter_region]:
                    population[iter_region, iter_year] += int(float(sheet.cell_value(iter_table, 3+iter_year)))
    return pd.DataFrame(population.T, columns = Region, index = year).div(10**8)

