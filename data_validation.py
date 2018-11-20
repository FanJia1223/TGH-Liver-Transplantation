# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import csv
import pandas as pd
df = pd.read_excel('testing.xlsx')

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
        return liveCount, liveRate, deathCount, deathRate

writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.

#root
liveCount, liveRate, deathCount, deathRate = getCountRate(df)
output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[91683, 563287, deathCount, deathCount],
                       'Rate':[0.14, 0.86, deathRate, liveRate]})
output.to_excel(writer, sheet_name='Sheet1', index=False)    

df1_L, df2_R = get_split(df, 'MELD', 26.5)
print("where OPOM predict live 91.26%.")

liveCount, liveRate, deathCount, deathRate = getCountRate(df1_L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[52009, 543332, deathCount, deathCount],
                       'Rate':[0.0874, 0.9126, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=6, startcol=0, index=False)

liveCount, liveRate, deathCount, deathRate = getCountRate(df1_L)

output = pd.DataFrame({'Model':['OPOM', '', 'data', ''], 
                       'Categories':['Death', 'Live', 'Death', 'Live'],
                       'Count':[52009, 543332, deathCount, deathCount],
                       'Rate':[0.0874, 0.9126, deathRate, liveRate]})
#output.to_excel(writer, sheet_name='Sheet1', index=False)
#output = pd.DataFrame({'model':[], 'liveRate' : [liveCount, liveRate, '']})
output.to_excel(writer, sheet_name='Sheet1', startrow=6, startcol=0, index=False)

writer.save()

#df2 = MELDSmallerThan(df1, 21.5)
#print("where OPOM predict live is 93.86%.")
#
#df3 = MELDSmallerThan(df2, 16.5)
#
#df4 = CalcRateSmallerThan(df3, 'Na', 133.2)
#print("where OPOM predict live is 90.08%.")
#
#df5 = CalcRateSmallerThan(df4, 'Na', 125.5)
#print("where OPOM predict live is 78.71%.")
#
#df6 = CalcRateSmallerThan(df5, 'Age', 65.23444)
#print("where OPOM predict live is 81.46%.")
#
#df7 = CalcRateSmallerThan(df6, 'ListYears', 0.437)
#print("where OPOM predict live is 73.72%.")
#
#df8 = CalcRateSmallerThan(df7, 'ALB', 2.25)
#print("where OPOM predict live is 52.50%. Terminal.")