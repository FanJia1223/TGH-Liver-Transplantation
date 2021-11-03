# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import csv
import pandas as pd
import numpy as np
from sklearn import metrics

df = pd.read_excel('Testing-Data-Jan-23.xlsx')

#def MELDSmallerThan ( df, num ):
#    newdf = df[['MELD', 'Live']]
#    print(newdf)
#    
#    live = sum(newdf['Live']) / len(newdf.index)
#    #print("Predict live: ", live)
#    if live >= 0.5:
#        print("Data live rate: " + "{:.2%}".format(live))
#    else: 
#        print("Data death rate: "+ "{:.2%}".format(live))
#        
#    newdf = df.loc[df['MELD'] < num]
#    print(newdf)
#    
#    return newdf
    
def get_split(df, feat, num):
    #newdf = df[[feat, 'Live']]
    #print(newdf)
    df_L = df.loc[df[feat] < num]
    df_R = df.loc[df[feat] >= num]
#    if newdf.empty:
#        print("Terminal node.")
#        return newdf

    #print(newdf)
    
    return df_L, df_R

def getCountRate (df):
        if len(df.index) != 0:
            liveCount = sum(df['Live'])
            liveRate = sum(df['Live']) / len(df.index)
            deathCount = len(df.index) - liveCount
            deathRate = 1 - liveRate
            #print("Predict live: ", live)
            print("Data live rate: " + "{:.2%}".format(liveRate))
        else:
            liveCount=0
            liveRate=0
            deathCount=0
            deathRate=0
        return liveCount, liveRate, deathCount, deathRate

writer = pd.ExcelWriter('confusion_matrix_output_Nov30.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.

#root
liveCount, liveRate, deathCount, deathRate = getCountRate(df)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
    
output.to_excel(writer, sheet_name='Sheet1', index=False)    

# Layer 1 _______________________________________________________________________________
df1_L, df1_R = get_split(df, 'MELD', 26.5)
print("Root split")

liveCount, liveRate, deathCount, deathRate = getCountRate(df1_L)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=6, startcol=0, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df1_R)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=6, startcol=5, index=False)

#Layer 2 ________________________________________________________________________________
print("Layer 2 Split 1")
df2_1L, df2_1R = get_split(df1_L, 'MELD', 21.5) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df2_1L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=12, startcol=0, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df2_1R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=12, startcol=5, index=False) # change row +6 and column +5 here

print("Layer 2 Split 2") # ................................................................
df2_2L, df2_2R = get_split(df1_R, 'BILT', 15.75) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df2_2L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=12, startcol=10, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df2_2R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=12, startcol=15, index=False) # change row +6 and column +5 here

#Layer 3 ________________________________________________________________________________
print("Layer 3 Split 1")
df3_1L, df3_1R = get_split(df2_1L, 'MELD', 16.5) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_1L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=0, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_1R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=5, index=False) # change row +6 and column +5 here

print("Layer 3 Split 2") # ................................................................
df3_2L, df3_2R = get_split(df2_1R, 'BILT', 3.345) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_2L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=10, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_2R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=15, index=False) # change row +6 and column +5 here

print("Layer 3 Split 3") # ................................................................
df3_3L, df3_3R = get_split(df2_2L, 'pBILT', 3.575) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_3L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=20, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_3R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=25, index=False) # change row +6 and column +5 here

print("Layer 3 Split 4") # ................................................................
df3_4L, df3_4R = get_split(df2_2R, 'Na', 144.5) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_4L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=30, index=False) #change row +6 here

liveCount, liveRate, deathCount, deathRate = getCountRate(df3_4R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=18, startcol=35, index=False) # change row +6 and column +5 here

# layer 4
print("Layer 4")
#---------------------------------------------------------------------------------
print ("Layer 4 - First Split ")
df4_1L, df4_1R = get_split(df3_1L, 'Na', 133.2)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_1L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[52009, 543332, deathCount, liveCount],
                       'Rate':[0.0874, 0.9126, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=0, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_1R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[52009, 543332, deathCount, liveCount],
                       'Rate':[0.0874, 0.9126, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=5, index=False)
#---------------------------------------------------------------------------------
print ("Layer 4 - Second Split ")
df4_2L, df4_2R = get_split(df3_1R, 'eINR', 0.1613)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_2L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[52009, 543332, deathCount, liveCount],
                       'Rate':[0.0874, 0.9126, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=10, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_2R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=15, index=False)
#---------------------------------------------------------------------------------
print ("Layer 4 - Third Split ")
df4_3L, df4_3R = get_split(df3_2L, 'Na', 132.9)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_3L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=20, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_3R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=25, index=False)


#---------------------------------------------------------------------------------
print ("Layer 4 - Fourth Split ")
df4_4L, df4_4R = get_split(df3_2R, 'Age', 54.92)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_4L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=30, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_4R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=35, index=False)
#---------------------------------------------------------------------------------
print ("Layer 4 - Fifth Split ")
df4_5L, df4_5R = get_split(df3_3L, 'eBILT', 0.1388)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_5L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=40, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_5R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=45, index=False)

#---------------------------------------------------------------------------------
print ("Layer 4 - Sixth Split ")
df4_6L, df4_6R = get_split(df3_3R, 'MELD', 32.5)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_6L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=50, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_6R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=55, index=False)

#---------------------------------------------------------------------------------
print ("Layer 4 - Seventh Split ")
df4_7L, df4_7R = get_split(df3_4L, 'Age', 48.38)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_7L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=60, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_7R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=65, index=False)

#---------------------------------------------------------------------------------
print ("Layer 4 - Eighth Split ")
df4_8L, df4_8R = get_split(df3_4R, 'Age', 50.48)


liveCount, liveRate, deathCount, deathRate = getCountRate(df4_8L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=70, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df4_8R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=24, startcol=75, index=False)

#Layer 5 ________________________________________________________________________________
print("Layer 5 Split 1")
df5_1L, df5_1R = get_split(df4_1L, 'Na', 125.2) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_1L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=0, index=False) #change row +6 here

death = np.array([1] * deathCount)  
live = np.array([0] * liveCount)
y_true = np.concatenate((death, live), axis = None)
scores = np.array([21.29] * (deathCount+liveCount))

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_1R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=5, index=False) # change row +6 and column +5 here
    
death = np.array([1] * deathCount)  
live = np.array([0] * liveCount)
y_true = np.concatenate((y_true, death, live), axis = None)
mort = np.array([9.09] * (deathCount+liveCount))
scores = np.concatenate((scores, mort), axis = None)

print("Layer 5 Split 2") # ................................................................
df5_2L, df5_2R = get_split(df4_1R, 'ALB', 2.605) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_2L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=10, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([6.97] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_2R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=15, index=False) # change row +6 and column +5 here
death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([2.65] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 3") # ................................................................
df5_3L, df5_3R = get_split(df4_2L, 'Na', 130.8) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_3L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=20, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([17.96] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_3R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=25, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([5.59] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 4") # ................................................................
df5_4L, df5_4R = get_split(df4_2R, 'dNa', 7.6) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_4L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=30, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([12.11] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_4R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=35, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([25.48] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 5") # ................................................................
df5_5L, df5_5R = get_split(df4_3L, 'pCREAT', 5.225) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_5L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=40, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([37.87] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_5R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=45, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([21.57] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 6") # ................................................................
df5_6L, df5_6R = get_split(df4_3R, 'ALB', 2.65) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_6L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=50, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([24.30] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_6R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=55, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([11.99] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 7") # ................................................................
df5_7L, df5_7R = get_split(df4_4L, 'Age', 43.09) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_7L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=60, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([18.68] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_7R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=65, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([28.04] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 8") # ................................................................
df5_8L, df5_8R = get_split(df4_4R, 'Na', 130.2) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_8L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=70, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([51.12] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_8R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=75, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([34.23] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 9") # ................................................................
df5_9L, df5_9R = get_split(df4_5L, 'Age', 24.05) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_9L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=80, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([0] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_9R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=85, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([23.93] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

print("Layer 5 Split 10") # ................................................................
df5_10L, df5_10R = get_split(df4_5R, 'CREAT', 1.785) # change variable, feature and value here

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_10L) # change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=90, index=False) #change row +6 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([40.92] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_10R) #change df here
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=95, index=False) # change row +6 and column +5 here

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([50.33] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

# layer 5
print("Layer 5")
#---------------------------------------------------------------------------------
print ("Layer 5 - 11th Split ")
df5_11L, df5_11R = get_split(df4_6L, 'ALB', 2.15)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_11L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=100, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([62.24] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_11R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=105, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([48.09] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

#---------------------------------------------------------------------------------
print ("Layer 5 - 12th Split ")
df5_12L, df5_12R = get_split(df4_6R, 'Age', 36.51)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_12L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=110, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([48.11] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_12R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=115, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([84.74] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

#---------------------------------------------------------------------------------
print ("Layer 5 - 13th Split ")
df5_13L, df5_13R = get_split(df4_7L, 'MELD', 30.5)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_13L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=120, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([42.07] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_13R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=125, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([83.66] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************


#---------------------------------------------------------------------------------
print ("Layer 5 - 14th Split ")
df5_14L, df5_14R = get_split(df4_7R, 'pBILT', 25.75)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_14L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=130, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([85.65] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_14R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=135, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([90.25] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************
#---------------------------------------------------------------------------------
print ("Layer 5 - 15th Split ")
df5_15L, df5_15R = get_split(df4_8L, 'INR', 2.63)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_15L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=140, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([83.48] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_15R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=145, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([99.08] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

#---------------------------------------------------------------------------------
print ("Layer 5 - 16th Split ")
df5_16L, df5_16R = get_split(df4_8R, 'INR', 1.905)

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_16L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=150, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([79.89] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

liveCount, liveRate, deathCount, deathRate = getCountRate(df5_16R)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[0, 0, deathCount, liveCount],
                       'Rate':[0, 0, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=30, startcol=155, index=False)

death = np.array([1] * deathCount)  # ****************************************************
live = np.array([0] * liveCount)   # ****************************************************
y_true = np.concatenate((y_true, death, live), axis = None)  # *************************************
mort = np.array([98.19] * (deathCount+liveCount))  # *************************************
scores = np.concatenate((scores, mort), axis = None) # *************************************

writer.save()


#fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
## method I: plt
#import matplotlib.pyplot as plt
#plt.title('ROC')
#plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()




df = pd.read_excel('MELDAUC.xlsx')

# =============================================================================
# y_true_meld = np.array(df['Final Status'])
# y_pred_meld = np.array(df['MELD_Prob'])
# 
# fpr, tpr, thresholds = metrics.roc_curve(y_true_meld, y_pred_meld, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('ROC')
# plt.plot(fpr, tpr,'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# =============================================================================


plt.figure(0).clf()
plt.title('ROC')
y_true_meld = np.array(df['Final Status'])
y_pred_meld = np.array(df['MELD_Prob'])
fpr_meld, tpr_meld, thresholds_meld = metrics.roc_curve(y_true_meld, y_pred_meld, pos_label=1)
auc_meld = metrics.roc_auc_score(y_true_meld, y_pred_meld)
plt.plot(fpr_meld,tpr_meld,label="MELD, AUC=%0.2f" % auc_meld)

y_true_na = np.array(df['Final Status'])
y_pred_na = np.array(df['MELDNA_Prob'])
fpr_na, tpr_na, thresh_na = metrics.roc_curve(y_true_na, y_pred_na, pos_label=1)
auc_na = metrics.roc_auc_score(y_true_na, y_pred_na)
plt.plot(fpr_na,tpr_na,label="MELD-Na, AUC=%0.2f" % auc_na)


fpr_OPOM, tpr_OPOM, thresholds_OPOM = metrics.roc_curve(y_true, scores, pos_label=1)
auc_OPOM = metrics.roc_auc_score(y_true, scores/100)
plt.plot(fpr_OPOM,tpr_OPOM,label="OPOM, AUC=%0.2f" % auc_OPOM)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






