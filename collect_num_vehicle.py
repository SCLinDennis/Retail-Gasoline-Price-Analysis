
# coding: utf-8

# In[8]:


import xlrd 
import numpy as np
  
# Give the location of the file 
loc = ("./2007.xls") 
  
# To open Workbook 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

dict_2007 = {}    
# For row 0 and column 0 
# states_list = []
# total_vehicles = []
new_states_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
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
# print(len(new_states_list))
for i in range(13,64):
#     print(sheet.cell_value(i, 13))
    dict_2007[new_states_list[i-13]] = sheet.cell_value(i, 13)

print(dict_2007)
#cmap = 'gray'


# In[3]:


loc = ("./{}.xls".format(2007))
print(loc)


# In[32]:


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
West_Coast = ('Washington', 'Oregon', 'California', 'Nevada', 'Arizona', 'Alaska', 'Hawaii')
Rocky_Mountain = ('Montana', 'Idaho', 'Wyoming', 'Utah', 'Colorado')
Gulf_Coast = ('New Mexico',  'Texas', 'Arkansas', 'Louisiana',  'Mississippi', 'Alabama')
Midwest = ('North Dakota',  'South Dakota', 'Nebraska',  'Kansas', 'Oklahoma', 
           'Minnesota', 'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 'Indiana', 'Kentucky',
           'Michigan',  'Tennessee', 'Ohio')

East_coast = ('Florida',  'Georgia', 'South Carolina', 'North Carolina', 'Virginia', 'West Virginia', 
              'Maryland',  'Delaware',  'Pennsylvania',  'New Jersey',  'New York',  'Connecticut', 
              'Rhode Island',  'Vermont',  'New Hampshire',  'Massachusetts',  'Maine', 'Dist. of Col.')

dic_vehicle = {West_Coast:0, Rocky_Mountain:0, Gulf_Coast:0, Midwest:0, East_coast:0}
      
        
loc = ("./2007.xls") 
  
# To open Workbook 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

dict_2007 = {}        

regions_list = [West_Coast, Rocky_Mountain, Gulf_Coast, Midwest, East_coast]

for iter_year in range(14,18):
    print(iter_year)
    loc = ("./{}.xlsx".format(iter_year+1997))
#     print(loc)
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
#     dic_'{}'.format(iter_year+1997) = {West_Coast:0, Rocky_Mountain:0, Gulf_Coast:0, Midwest:0, East_coast:0}
    temp_dict = {West_Coast:0, Rocky_Mountain:0, Gulf_Coast:0, Midwest:0, East_coast:0}
    left = 12
    right = 63
    coloumn_total = 15
    for iter_table in range(left, right):
        for key, value in temp_dict.items():    
            if All_states[iter_table-left] in key:
#                 print('lalala')
#                 print('nothing?', sheet.cell_value(iter_table, 13))
                temp_dict[key] += int(float(sheet.cell_value(iter_table, coloumn_total)))

#     'dic_{}'.format(iter_year+1997) = temp_dict
    np.save("dic_{}.npy".format(iter_year+1997), temp_dict)

print(temp_dict[regions_list[0]])
        


# In[35]:


# dic_1997 = np.load('dic_2010.npy').item()
# print(dic_1997[regions_list[2]]) # displays "world"
regions_list = [West_Coast, Rocky_Mountain, Gulf_Coast, Midwest, East_coast]
sum_test = 0
for year_i in range(15, 16):#there is 18 years file
    npy_file = "dic_{}.npy".format(year_i + 1997)
    temp_dict = np.load(npy_file).item()
    for region_i in range(len(regions_list)):
        sum_test += temp_dict[regions_list[region_i]]

print(sum_test)

