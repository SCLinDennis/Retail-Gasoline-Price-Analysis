"""
Created on Mon Feb 11 12:34:00 2019
@author: DennisLin
"""
#https://docs.google.com/document/d/14yTbyoFJks_aDMtJLcwqm7MG5n-DEk_aJzrVTy8EJUA/edit
import matplotlib
matplotlib.use('TkAgg')   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
#
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
#    value_array = np.array(value_data)
    array = np.array([date_data, value_data])
#    return np.hstack((date_array, value_array))
    return array

def dict2df(dic):
    df = pd.DataFrame()
    for name, url in dic.items():
        df = pd.concat([df, url2df(url, name)], axis = 1)
    return df

def dict2df_im(dic):
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
    max_ = np.max(numpydata)
    min_ = np.min(numpydata)
    return (numpydata-min_)/(max_-min_)
#%%
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
if __name__ == "__main__":
    gas_price = {}
    price_dict, import_dict, stock_dict, export_dict, refine_dict  = {}, {}, {}, {}, {}

    #store url
    for i, r in enumerate(Region):
        price_dict[r] = price_url.replace('10', '%d') %(10*(i+1))
        import_dict[r] = import_url.replace('10', '%d') %(10*(i+1))
        stock_dict[r] = stock_url.replace('11', '%d') %(10*(i+1) + 1)
#        export_dict[r] = export_url.replace('11', '%d') %(10*(i+1) + 1)
#        refine_dict[r] = refine_url.replace('12', '%d') %(10*(i+1) + 2)

    #load url
    df_price = dict2df(price_dict).div(1000)
    df_import = dict2df_im(import_dict)
    df_stock = dict2df(stock_dict)

    

    #plot figure
    price_fig = df_price.plot.line().get_figure()#1993~Apr, 2019~Feb
    import_fig = df_import.plot.line().get_figure()
#    import_fig.xticks(df_import.index.values)
#    get_figure()#2004~Apr, 2019~Feb
    stock_fig = df_stock.plot.line().get_figure()#1990~Jan, 2019~Feb.
    
    
    #save figure    
    price_fig.savefig('price.png')
    import_fig.savefig('import.png')
    stock_fig.savefig('stock.png')
    
    import_cor = get_correlate(df_price, df_import)
    stock_cor = get_correlate(df_price, df_stock)
   
    
#%%
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)

import_corr_map = {}
for i, region in enumerate(state_regions):
    for state in region:
        import_corr_map[state] = import_cor[Region[i]][0]

colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
#cmap=plt.cm.reds
#vmin = 0.1; vmax = 0.35 # set range.
vmin = 0.1; vmax = 0.35 

ATOLL_CUTOFF = 0.005 #delete tiny islands(area<0.005) of Hawaii 
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = import_corr_map[statename]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
#        colors[statename] = cmap(1.-((pop-vmin)/(vmax-vmin))**0.5)[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    #fill the color for each states
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
#for nshape, seg in enumerate(m.states):
#    # skip DC and Puerto Rico.
#    if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
#    # Offset Alaska and Hawaii to the lower-left corner. 
#        if statenames[nshape] == 'Alaska':
#        # Alaska is too big. Scale it down to 35% first, then transate it. 
#            seg = list(map(lambda xy: (0.20*xy[0] + 700000, 0.35*xy[1]-1300000), seg))
#        if statenames[nshape] == 'Hawaii':
#            seg = list(map(lambda xy: (xy[0] + 5100000, xy[1]-1500000), seg))
#
#        color = rgb2hex(colors[statenames[nshape]]) 
#        poly = Polygon(seg,facecolor=color,edgecolor='black')
#        ax.add_patch(poly)
plt.title('Correlation Heatmap (price vs. import)')
#plt.colorbar(ax)
plt.show()    

#%%
import numpy as np
import matplotlib.pyplot as plt

region_name,imp_value,stk_value=[],[],[]
for reg, val in import_cor.items():
    region_name.append(reg)
    imp_value.append(round(val[0],3))
for reg, val in stock_cor.items():
#    region_name.append(reg)
    stk_value.append(round(val[0],3))
imp=tuple(imp_value)
stk=tuple(stk_value)
region=tuple(region_name)

#men_std = (2, 3, 4, 1, 2)
#women_std = (3, 5, 2, 3, 3)

ind = np.arange(len(imp))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, imp, width, 
                color='SkyBlue', label='import vs. price')
rects2 = ax.bar(ind + width/2, stk, width,
                color='IndianRed', label='stock vs. price')

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
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()