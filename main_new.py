#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:34:00 2019

@author: DennisLin
"""
#https://docs.google.com/document/d/14yTbyoFJks_aDMtJLcwqm7MG5n-DEk_aJzrVTy8EJUA/edit
import xlrd
import matplotlib
matplotlib.use('TkAgg')   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from pyquery import PyQuery as pq
#%%

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

def url2df_im(url, name):
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
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df_im(url, name)], axis = 1)
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
    '''
    Normalized the numpy array.
    '''
    max_ = np.max(numpydata)
    min_ = np.min(numpydata)
    return (numpydata-min_)/(max_-min_)

def name_process(array):
    '''
    preprocess the naming conflict between two object
    '''
    out = []
    for i, region in enumerate(state_regions):
        out.append(array.item().get(region))
    return out

def create_heatmap(coor_dict, cmap, tit):
    '''
    Create the heapmap of correlation between factors and the fossil prices.
    For more color map, please check: https://matplotlib.org/examples/color/colormaps_reference.html
    Args:
        coor_dict: (dict)
        cmap: the colormap type
        title: (str) i.e. 'Correlation between price and import in different region'
    '''
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    m.readshapefile('./shape/st99_d00','states',drawbounds=True)
    
    colors={}
    statenames=[]
    ATOLL_CUTOFF=0.005
    vmin = 0; vmax = 1
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        if statename not in ['District of Columbia','Puerto Rico']:
            pop = coor_dict[statename]
            colors[statename] = cmap(np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
        statenames.append(statename)
    ax = plt.gca() # get current axes instance
    fig = plt.gcf()
   # 
    for nshape,seg in enumerate(m.states):
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            color = rgb2hex(colors[statenames[nshape]]) 
            poly = Polygon(seg,facecolor=color,edgecolor='black')
            ax.add_patch(poly)
    for i, shapedict in enumerate(m.states_info):
    #fill the color for hawaii and Alaska
        if shapedict['NAME'] not in ['Puerto Rico', 'District of Columbia']:
    # Translate the noncontiguous states:
            if shapedict['NAME'] in ['Alaska', 'Hawaii']:
                seg = m.states[int(shapedict['SHAPENUM'] - 1)]
                # maintain the information of 8 main islands of Hawaii, rescale
                if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > ATOLL_CUTOFF:
                    seg = list(map(lambda xy: ((xy[0] + 5500000)*0.8, 0.8*(xy[1]-1200000)), seg))
                # Rescale Alaska
                elif shapedict['NAME'] == 'Alaska':
                    seg = list(map(lambda xy: (0.33*xy[0] + 1100000, 0.33*xy[1]-1300000), seg))
        
            color = rgb2hex(colors[shapedict['NAME']]) 
            poly = Polygon(seg, facecolor=color, edgecolor='black', linewidth=0.8)
            ax.add_patch(poly)
        #
#    for nshape, seg in enumerate(m.states):
#        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
#            if statenames[nshape] == 'Alaska':
#                seg = list(map(lambda xy: (0.20*xy[0] + 700000, 0.35*xy[1]-1300000), seg))
#            if statenames[nshape] == 'Hawaii':
#                seg = list(map(lambda xy: (xy[0] + 5100000, xy[1]-1500000), seg))
#    
#            color = rgb2hex(colors[statenames[nshape]]) 
#            poly = Polygon(seg,facecolor=color,edgecolor=color)
#            ax.add_patch(poly)
    plt.title(tit)
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05]) # posititon
    cb = ColorbarBase(cax,cmap=cmap, orientation='horizontal')
    plt.show() 

def create_corr_map(corr_dict):
    '''
    create the dictionary suitable for mapping to the US map
    Args:
        corr_dict: (dictionary) 
    Return:
        corr_map: (dictionary)
    '''
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
    df.index = pd.to_datetime(df.index)
    df_tmp = df_price.groupby(df.index.year).transform('mean')
    df_price_down = df_tmp.iloc[(df_tmp.index.month == 2) & (df_tmp.index.year >= start_year) & (df_tmp.index.year <= end_year)]
    return df_price_down

def corr_bar(import_cor,vehicle_cor,pop_cor):
    region_name,imp_value,vhc_value,pop_value=[],[],[],[]
    for reg, val in import_cor.items():
        region_name.append(reg)
        imp_value.append(round(val[0],3))
    for reg, val in vehicle_cor.items():
#        region_name.append(reg)
        vhc_value.append(round(val[0],3))
    for reg, val in pop_cor.items():
#        region_name.append(reg)
        pop_value.append(round(val[0],3))
    imp=tuple(imp_value)
    vhc=tuple(vhc_value)
    pop=tuple(pop_value)
    region=tuple(region_name)
    
    ind = np.arange(len(imp))  # the x locations for the groups
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width, imp, width, 
                    color='SkyBlue', label='import vs. price')
    rects2 = ax.bar(ind , vhc, width,
                    color='IndianRed', label='vehicle vs. price')
    rects3 = ax.bar(ind + width, pop, width,
                    color='Purple', label='population vs. price')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation among different factors (5 regions)')
    ax.set_xticks(ind)
    ax.set_xticklabels(region)
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
    plt.show()
#%%
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
Region = ['East', 'Midwest', 'GC', 'RM', 'WC']
price_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPM0_PTE_R10_DPG&f=W'
import_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WTTIM_R10-Z00_2&f=W'
stock_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTP11&f=W'
#export_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTEXP11&f=M'
#refine_url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCRRIP12&f=W'#crude oil input
num_year = 18
#%%
if __name__ == "__main__":
    gas_price = {}
    price_dict, import_dict, stock_dict, export_dict, refine_dict  = {}, {}, {}, {}, {}

    #store url
    for i, r in enumerate(Region):
        price_dict[r] = price_url.replace('10', '%d') %(10*(i+1))
        import_dict[r] = import_url.replace('10', '%d') %(10*(i+1))
#        stock_dict[r] = stock_url.replace('11', '%d') %(10*(i+1) + 1)

    #load url
    df_price = dict2df(price_dict).div(1000)
    df_import = dict2df_im(import_dict)
#    df_stock = dict2df(stock_dict)

    #plot figure
    
    # price_fig = df_price.rolling(12).mean().plot.line().get_figure()#1993~Apr, 2019~Feb
    price_fig = df_price.rolling(12).mean().plot.line(title= 'Retail Gas Price v.s. Time')
    price_fig.set_xlabel("Year")
    price_fig.set_ylabel("Gas Price ($)")
    plt.show()
    import_fig = df_import.rolling(12).mean().plot.line(title = 'Import of Crude Oil v.s. Time')
    import_fig.set_xlabel("Year")
    import_fig.set_ylabel("Barrels per Day (k)")
    plt.show() 
    #save figure    
    # price_fig.savefig('price.png')
    # import_fig.savefig('import.png')
    
    import_cor = get_correlate(df_price, df_import)
    print('The correlation of import:', import_cor)
    #%% 
    import_corr_map = create_corr_map(import_cor)
    title = 'Correlation: Price v.s. Import'
    create_heatmap(import_corr_map, plt.cm.PiYG, title)

    #%%
    #load the vehicles data
    year = []
    for year_i in range(num_year):
        year.append(str(1997+year_i))
        npy_file = "./vehicle/dic_{}.npy".format(year_i + 1997)
        tmp_array = name_process(np.load(npy_file))
        new_vehicle = np.array(tmp_array).reshape([1, -1]) 
        if year_i == 0:
            vehicles = new_vehicle
        else:
            vehicles = np.vstack((vehicles, new_vehicle))
    df_vehicle = pd.DataFrame(vehicles, columns = [Region], index = year)
    ve_fig = df_vehicle.plot.line(title = 'Vehicles v.s. Time')
    ve_fig.set_xlabel("Year")
    ve_fig.set_ylabel("Vehicles")    
    plt.show()    
    df_price_down = downsample(df_price, int(df_vehicle.index[0]), int(df_vehicle.index[-1]))
    vehicle_cor = get_correlate(df_price_down, df_vehicle)    
    print('The correlation of vehicles:', vehicle_cor)
    #build the map
    vehicle_corr_map = create_corr_map(vehicle_cor)
    title = 'Correlation: Price v.s. Vehicles'
    create_heatmap(vehicle_corr_map, plt.cm.coolwarm, title)    
    
    #%%
    #load the populations data
    
    loc_pop = ("./population/nst-est2018-01.xlsx") 
    wb = xlrd.open_workbook(loc_pop) 
    sheet = wb.sheet_by_index(0) 
    population = np.zeros((5, 9))    
    left = 9
    right = 60
    coloumn_total = 15
    year = []
    for iter_year in range(9):
        year.append(2010+iter_year)
        for iter_table in range(left, right):
            for iter_region in range(5):
                if All_states[iter_table-left] in state_regions[iter_region]:
                    population[iter_region, iter_year] += int(float(sheet.cell_value(iter_table, 3+iter_year)))

    df_population = pd.DataFrame(population.T, columns = [Region], index = year)
    pop_fig = df_population.plot.line(title = 'Population v.s. Time')
    pop_fig.set_xlabel("Year")
    pop_fig.set_ylabel("Population")
    plt.show()    
    df_price_down = downsample(df_price, int(df_population.index[0]), int(df_population.index[-1]))

    #get correlation
    pop_cor = get_correlate(df_price_down, df_population) 
    print('The correlation of populations:', pop_cor)
    #build the map
    pop_corr_map = create_corr_map(pop_cor)
    title = 'Correlation: price v.s. population'
    create_heatmap(pop_corr_map, plt.cm.BrBG, title)    

    #%%
    bar_chart=corr_bar(import_cor,vehicle_cor,pop_cor)