'''
This part is to collect the amount of vehicles in 5 regions from 1997 to 2014
'''
# the states:
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

# the states in 5 regions
West_Coast = ('Washington', 'Oregon', 'California', 'Nevada', 'Arizona', 'Alaska', 'Hawaii')
Rocky_Mountain = ('Montana', 'Idaho', 'Wyoming', 'Utah', 'Colorado')
Gulf_Coast = ('New Mexico',  'Texas', 'Arkansas', 'Louisiana',  'Mississippi', 'Alabama')
Midwest = ('North Dakota',  'South Dakota', 'Nebraska',  'Kansas', 'Oklahoma', 
           'Minnesota', 'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 'Indiana', 'Kentucky',
           'Michigan',  'Tennessee', 'Ohio')

East_coast = ('Florida',  'Georgia', 'South Carolina', 'North Carolina', 'Virginia', 'West Virginia', 
              'Maryland',  'Delaware',  'Pennsylvania',  'New Jersey',  'New York',  'Connecticut', 
              'Rhode Island',  'Vermont',  'New Hampshire',  'Massachusetts',  'Maine', 'Dist. of Col.')     

regions_list = [East_coast, Midwest, Gulf_Coast, Rocky_Mountain, West_Coast]
Region = ['East', 'Midwest', 'GC', 'RM', 'WC']
dict_list_year = []

# collect the amount of the vehicles in 5 regions from 1997 to 1999
# save them in a dictionary

for iter_year in range(0,3): 
    loc = ("./{}.xls".format(iter_year+1997))
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    temp_dict = {Region[0]:0, Region[1]:0, Region[2]:0, Region[3]:0, Region[4]:0}
    left = 14
    right = left+51
    coloumn_total = 13 
    for iter_table in range(left, right):
        for iter_region in range(5):
            if All_states[iter_table-left] in regions_list[iter_region]:
                temp_dict[Region[iter_region]] += int(float(sheet.cell_value(iter_table, coloumn_total)))
    dict_list_year.append(temp_dict)

# collect the amount of the vehicles in 5 regions from 2000 to 2010
# save them in a dictionary

for iter_year in range(3,14): 
    loc = ("./{}.xls".format(iter_year+1997))
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    temp_dict = {Region[0]:0, Region[1]:0, Region[2]:0, Region[3]:0, Region[4]:0}
    left = 13
    right = left+51
    coloumn_total = 13 #15
    for iter_table in range(left, right):
        for iter_region in range(5):
            if All_states[iter_table-left] in regions_list[iter_region]:

                temp_dict[Region[iter_region]] += int(float(sheet.cell_value(iter_table, coloumn_total)))
    dict_list_year.append(temp_dict)

# collect the amount of the vehicles in 5 regions from 2011 to 2014
# save them in a dictionary

for iter_year in range(14,18): # 14, 18
    loc = ("./{}.xlsx".format(iter_year+1997))
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    temp_dict = {Region[0]:0, Region[1]:0, Region[2]:0, Region[3]:0, Region[4]:0}
    left = 12
    right = left+51
    coloumn_total = 15 #15
    for iter_table in range(left, right):
        for iter_region in range(5):
            if All_states[iter_table-left] in regions_list[iter_region]:

                temp_dict[Region[iter_region]] += int(float(sheet.cell_value(iter_table, coloumn_total)))
    dict_list_year.append(temp_dict)    

import numpy as np

'''
extract population data from the raw xlsx file, and save them in a list
'''
loc_pop = ("./nst-est2018-01.xlsx") 
wb = xlrd.open_workbook(loc_pop) 
sheet = wb.sheet_by_index(0) 
pop_matrix = np.zeros((5, 9))

regions_list = [East_coast, Midwest, Gulf_Coast, Rocky_Mountain, West_Coast]

left = 9
right = 60
coloumn_total = 15
for iter_year in range(9):
    for iter_table in range(left, right):
        for iter_region in range(5):
            if All_states[iter_table-left] in regions_list[iter_region]:
                pop_matrix[iter_region, iter_year] += int(float(sheet.cell_value(iter_table, 3+iter_year)))

print(pop_matrix)



