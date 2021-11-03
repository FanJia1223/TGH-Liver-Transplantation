# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:23:35 2018

@author: jiafan
"""

import pandas as pd
import numpy as np
from sklearn import metrics 
import seaborn as sns 
sns.set()
import matplotlib.pyplot as plt

df = pd.read_excel('Cleaned Master File 2018-12-17.xlsx')
df.head()

# df.count() does not include NaN values
print(df.info())
print(df.describe())





 

# Missing Values 

 # Heatmap
sns.heatmap(df.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')
plt.title('Missing Data')
plt.show()

#For now we can remove useless columns 
df_val = df
df_val.drop(['BW_1mo','BW_2mo','BW_3mo','BW_4mo','BW_5mo','BW_6mo','BW_7mo','BW_8mo','BW_9mo',
             'BW_10mo','BW_11mo','BW_1yr','BW_>1yr','Unlisted_Lab_Int','Identifier','Identifier.1','Identifier'], inplace=True, axis=1, errors='ignore')
df_val.drop(['SUBJECT_ID.2','LISTING_DATE.2','DATE_OF_LAB.1','Identifier','BW At Listing','SUBJECT_ID.1','LISTING_DATE.1'],inplace=True, axis=1, errors='ignore')
df_val.drop(['DIALYSIS_TYPE_DESCR'], inplace=True, axis=1, errors='ignore')




#Remove columns with some missing values 

# Have to choose EST_GFR or HCO3','MG','PHOS','CALC'
df_val.drop(['DATE_OF_TRANSPLANT','AGE_AT_UNLISTING','AGE_AT_DEATH','DATE_OF_LAB','LAB_INTERVAL',
             'Identifier','LAB_INTERVAL.1'], inplace=True, axis=1, errors='ignore')
df_num = df_val.select_dtypes(include = ['float64', 'int64'])
df_val.drop(['EST_GFR'], inplace=True, axis=1, errors='ignore')

# Remove rows with missing data
#df_3 = df_num.dropna(how='any',subset = ['HCO3','MG','PHOS','CALC','GLUCOSE','ALB','MELD','ALK_PHOS'])
df_4 = df_val.dropna(how='any')
#df_1 = df_num.dropna(how='any',subset = ['EST_GFR'])
# We cannont include all the attributes as there is only 1 entry left if we want all attributes 
#df_all = df_num.dropna(how='any')
#df_2 = df_num.dropna(how='any',subset = ['HCO3','MG','PHOS','CALC'])

df_4.info()

# Use df_4 for EDA 

#Category Features 
#Remove unnecessary columns 
df_4.drop(['SUBJECT_ID','LISTING_DATE','TX_NUMBER','DIALYSIS_TYPE','Identifier','Final State'], inplace=True, axis=1, errors='ignore')
df_4.describe().transpose()


### Finally EDA KAISHILE 
# Survival Count
print('Target Variable')
print(df_4.groupby(['Drop_3M']).Drop_3M.count())

# Target Variable Countplot
sns.set_style('darkgrid')
plt.figure(figsize = (10,5))
sns.countplot(df_4['Drop_3M'], alpha =.80, palette= ['lightblue','lightgreen'])
plt.title('Live vs Drof_off in 3 Months')
plt.ylabel('# Patients')
plt.show()

#Numeric Features 
# Identify numeric features
print('Continuous Variables')
print(df_4[['CREAT',	'BILT',	'AST',	'ALT',	'ALK_PHOS',	'PLATELETS',	'WBC',	'ALB',	'MELD',	'HGB',	'INR',	'TREATMENT_PHASE',	'Na',	'HCO3',	'GLUCOSE',	'CALC',	'MG',	'PHOS',	'Drop_3M']].describe().transpose())

print('--'*40)
print('Discrete Variables')
print(df_4.groupby('PATIENT_STATUS_DESCR').PATIENT_STATUS_DESCR.count())
print(df_4.groupby('TREATMENT_PHASE_DESCR').TREATMENT_PHASE_DESCR.count())
print(df_4.groupby('SEX_DESCR').SEX_DESCR.count())
print(df_4.groupby('RACE_DESCR').RACE_DESCR.count())
print(df_4.groupby('BLOOD_TYPE_DESCR').BLOOD_TYPE_DESCR.count())
print(df_4.groupby('RH_FACTOR_DESCR').RH_FACTOR_DESCR.count())
print(df_4.groupby('P_DIAG_DESCR').P_DIAG_DESCR.count())
print(df_4.groupby('SEC_DIAG_DESCR').SEC_DIAG_DESCR.count())
print(df_4.groupby('SMOKER').SMOKER.count())
print(df_4.groupby('PREV_SURG1').PREV_SURG1.count())
print(df_4.groupby('LIFE_SUPPORT').LIFE_SUPPORT.count())
print(df_4.groupby('VENTILATOR_DEPENDENT').VENTILATOR_DEPENDENT.count())
print(df_4.groupby('IV_INATROPES').IV_INATROPES.count())
print(df_4.groupby('VAD').VAD.count())
print(df_4.groupby('IABP').IABP.count())
print(df_4.groupby('TIPS').TIPS.count())
print(df_4.groupby('ENCEPHALOPATHY').ENCEPHALOPATHY.count())
print(df_4.groupby('VARICES').VARICES.count())
print(df_4.groupby('ASCITES').ASCITES.count())
print(df_4.groupby('UNLISTED_REASON').UNLISTED_REASON.count())
print(df_4.groupby('UNLISTED_REASON_DESCR').UNLISTED_REASON_DESCR.count())
print(df_4.groupby('CAUSE_OF_DEATH').CAUSE_OF_DEATH.count())
print(df_4.groupby('CAUSE_OF_DEATH_DESCR').CAUSE_OF_DEATH_DESCR.count())


#Drop somemore useless attributes
df_4.drop(['ENCEPHALOPATHY','VARICES','ASCITES'], inplace=True, axis=1, errors='ignore')

# pdf graphs
fig = plt.figure(figsize = (16,10))
fig.subplots_adjust(hspace = .30)
ax1 = fig.add_subplot(221)
ax1.hist(df_4[df_4['Drop_3M'] ==0].CREAT, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax1.hist(df_4[df_4['Drop_3M']==1].CREAT, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'grey')
ax1.set_title('CREAT: Live vs Drop_off')
ax1.set_xlabel('CREAT')
ax1.set_ylabel('probability density')
ax1.legend(loc = 'upper right')

ax2 = fig.add_subplot(222)
ax2.hist(df_4[df_4['Drop_3M'] ==0].BILT, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax2.hist(df_4[df_4['Drop_3M']==1].BILT, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'grey')
ax2.set_title('BILT: Live vs Drop_off')
ax2.set_xlabel('BILT')
ax2.set_ylabel('probablity density')
ax2.legend(loc = 'upper right')

#sns.boxplot(x="Drop_3M", y="CREAT", data=df_4)
#sns.violinplot(x="Drop_3M", y="CREAT", data=df_4)

ax3 = fig.add_subplot(223)
ax3.hist(df_4[df_4['Drop_3M'] ==0].AST, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax3.hist(df_4[df_4['Drop_3M']==1].AST, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'grey')
ax3.set_title('AST: Live vs Drop_off')
ax3.set_xlabel('AST')
ax3.set_ylabel('probability density')
ax3.legend(loc = 'upper right')


ax4 = fig.add_subplot(224)
ax4.hist(df_4[df_4['Drop_3M'] ==0].ALT, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax4.hist(df_4[df_4['Drop_3M']==1].ALT, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'grey')
ax4.set_title('ALT: Live vs Drop_off')
ax4.set_xlabel('ALT')
ax4.set_ylabel('probability density')
ax4.legend(loc = 'upper right')


plt.show()
#---------------------------------------------------------------------------------------------------------------------
fig2 = plt.figure(figsize = (16,10))
fig2.subplots_adjust(hspace = .30)
ax1 = fig2.add_subplot(221)
ax1.hist(df_4[df_4['Drop_3M'] ==0].ALK_PHOS, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax1.hist(df_4[df_4['Drop_3M']==1].ALK_PHOS, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'grey')
ax1.set_title('ALK_PHOS: Live vs Drop_off')
ax1.set_xlabel('ALK_PHOS')
ax1.set_ylabel('probability density')
ax1.legend(loc = 'upper right')

ax2 = fig2.add_subplot(222)
ax2.hist(df_4[df_4['Drop_3M'] ==0].PLATELETS, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax2.hist(df_4[df_4['Drop_3M']==1].PLATELETS, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax2.set_title('PLATELETS: Live vs Drop_off')
ax2.set_xlabel('PLATELETS')
ax2.set_ylabel('probability density')
ax2.legend(loc = 'upper right')

#sns.boxplot(x="Drop_3M", y="CREAT", data=df_4)
#sns.violinplot(x="Drop_3M", y="CREAT", data=df_4)

ax3 = fig2.add_subplot(223)
ax3.hist(df_4[df_4['Drop_3M'] ==0].WBC, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax3.hist(df_4[df_4['Drop_3M']==1].WBC, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax3.set_title('WBC: Live vs Drop_off')
ax3.set_xlabel('WBC')
ax3.set_ylabel('probability density')
ax3.legend(loc = 'upper right')


ax4 = fig2.add_subplot(224)
ax4.hist(df_4[df_4['Drop_3M'] ==0].ALB, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax4.hist(df_4[df_4['Drop_3M']==1].ALB, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax4.set_title('ALB: Live vs Drop_off')
ax4.set_xlabel('ALB')
ax4.set_ylabel('probability density')
ax4.legend(loc = 'upper right')


plt.show()
#-----------------------------------------------------------------------------------------------------------------------
fig3 = plt.figure(figsize = (16,10))
fig3.subplots_adjust(hspace = .30)
ax1 = fig3.add_subplot(221)
ax1.hist(df_4[df_4['Drop_3M'] ==0].MELD, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax1.hist(df_4[df_4['Drop_3M']==1].MELD, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax1.set_title('MELD: Live vs Drop_off')
ax1.set_xlabel('MELD')
ax1.set_ylabel('probability density')
ax1.legend(loc = 'upper right')

ax2 = fig3.add_subplot(222)
ax2.hist(df_4[df_4['Drop_3M'] ==0].HGB, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax2.hist(df_4[df_4['Drop_3M']==1].HGB, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax2.set_title('HGB: Live vs Drop_off')
ax2.set_xlabel('HGB')
ax2.set_ylabel('probability density')
ax2.legend(loc = 'upper right')

#sns.boxplot(x="Drop_3M", y="CREAT", data=df_4)
#sns.violinplot(x="Drop_3M", y="CREAT", data=df_4)

ax3 = fig3.add_subplot(223)
ax3.hist(df_4[df_4['Drop_3M'] ==0].INR, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax3.hist(df_4[df_4['Drop_3M']==1].INR, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax3.set_title('INR: Live vs Drop_off')
ax3.set_xlabel('INR')
ax3.set_ylabel('probability density')
ax3.legend(loc = 'upper right')


ax4 = fig3.add_subplot(224)
ax4.hist(df_4[df_4['Drop_3M'] ==0].AGE_AT_LISTING, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax4.hist(df_4[df_4['Drop_3M']==1].AGE_AT_LISTING, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax4.set_title('AGE_AT_LISTING: Live vs Drop_off')
ax4.set_xlabel('AGE_AT_LISTING')
ax4.set_ylabel('probability density')
ax4.legend(loc = 'upper right')


plt.show()
#-------------------------------------------------------------------------------------------------
fig4 = plt.figure(figsize = (16,10))
fig4.subplots_adjust(hspace = .30)
ax1 = fig4.add_subplot(221)
ax1.hist(df_4[df_4['Drop_3M'] ==0].Na, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax1.hist(df_4[df_4['Drop_3M']==1].Na, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax1.set_title('Na: Live vs Drop_off')
ax1.set_xlabel('Na')
ax1.set_ylabel('probability density')
ax1.legend(loc = 'upper right')

ax2 = fig4.add_subplot(222)
ax2.hist(df_4[df_4['Drop_3M'] ==0].HCO3, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax2.hist(df_4[df_4['Drop_3M']==1].HCO3, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax2.set_title('HCO3: Live vs Drop_off')
ax2.set_xlabel('HCO3')
ax2.set_ylabel('probability density')
ax2.legend(loc = 'upper right')

#sns.boxplot(x="Drop_3M", y="CREAT", data=df_4)
#sns.violinplot(x="Drop_3M", y="CREAT", data=df_4)

ax3 = fig4.add_subplot(223)
ax3.hist(df_4[df_4['Drop_3M'] ==0].GLUCOSE, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax3.hist(df_4[df_4['Drop_3M']==1].GLUCOSE, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax3.set_title('GLUCOSE: Live vs Drop_off')
ax3.set_xlabel('GLUCOSE')
ax3.set_ylabel('probability density')
ax3.legend(loc = 'upper right')


ax4 = fig4.add_subplot(224)
ax4.hist(df_4[df_4['Drop_3M'] ==0].CALC, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax4.hist(df_4[df_4['Drop_3M']==1].CALC, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax4.set_title('CALC: Live vs Drop_off')
ax4.set_xlabel('CALC')
ax4.set_ylabel('probability density')
ax4.legend(loc = 'upper right')


plt.show()
#-------------------------------------------------------------------------------------------------
fig5 = plt.figure(figsize = (16,10))
fig5.subplots_adjust(hspace = .30)

ax1 = fig5.add_subplot(221)
ax1.hist(df_4[df_4['Drop_3M'] ==0].MG, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax1.hist(df_4[df_4['Drop_3M']==1].MG, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax1.set_title('MG: Live vs Drop_off')
ax1.set_xlabel('MG')
ax1.set_ylabel('probability density')
ax1.legend(loc = 'upper right')

ax2 = fig5.add_subplot(222)
ax2.hist(df_4[df_4['Drop_3M'] ==0].PHOS, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax2.hist(df_4[df_4['Drop_3M']==1].PHOS, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax2.set_title('PHOS: Live vs Drop_off')
ax2.set_xlabel('PHOS')
ax2.set_ylabel('probability density')
ax2.legend(loc = 'upper right')
plt.show()

ax3 = fig5.add_subplot(223)
ax3.hist(df_4[df_4['Drop_3M'] ==0].AGE_AT_LISTING, density=True, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
ax3.hist(df_4[df_4['Drop_3M']==1].AGE_AT_LISTING, density=True, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
ax3.set_title('AGE_AT_LISTING: Live vs Drop_off')
ax3.set_xlabel('AGE_AT_LISTING')
ax3.set_ylabel('probability density')
ax3.legend(loc = 'upper right')

# =============================================================================
# ax4 = fig5.add_subplot(224)
# ax4.hist(df_4[df_4['Drop_3M'] ==0].AGE_AT_LISTING, bins = 25, label ='Live', alpha = .50,edgecolor= 'black',color ='lightgreen')
# ax4.hist(df_4[df_4['Drop_3M']==1].AGE_AT_LISTING, bins = 25, label = 'Drop_off', alpha = .50, edgecolor = 'black',color = 'blue')
# ax4.set_title('AGE_AT_LISTING: Live vs Drop_off')
# ax4.set_xlabel('AGE_AT_LISTING')
# ax4.set_ylabel('# Bloodtests')
# ax4.legend(loc = 'upper right')
# =============================================================================


plt.show()
#------------------------------------------------------------
# Passenger class summary

print(df_4.groupby(['SMOKER','Drop_3M']).SMOKER.count().unstack())

# Passenger class visualization
pclass = df_4.groupby(['SMOKER','Drop_3M']).SMOKER.count().unstack()
p1 = pclass.plot(kind = 'bar', stacked = True, 
                   title = 'Patient by smoking: Live & Drop_off', 
                   color = ['lightgreen','grey'], alpha = .70)
p1.set_xlabel('SMOKER')
p1.set_ylabel('# Bloodtests')
p1.legend(['Live','Drop_off'])
plt.show()

#-------------------------------------------------------------------------
sns.set_style('darkgrid')
f, axes = plt.subplots(1,2, figsize = (25,5))

# Plot [0]
sns.countplot(x = 'SEX_DESCR', data = df_4, palette = 'GnBu_d', ax = axes[0])
axes[0].set_xlabel('SEX_DESCR')
axes[0].set_ylabel('# Bloodtests')
axes[0].set_title('Gender of Patients')

# Plot [1]
sns.countplot(x = 'PATIENT_STATUS_DESCR', data = df_4, palette = 'GnBu_d',ax = axes[1])
axes[1].set_xlabel('PATIENT_STATUS_DESCR')
axes[1].set_ylabel('# Bloodtests')
axes[1].set_title('Patient Status')

#----------------------------------------------------------------------------
df_4num = df_4.select_dtypes(include = ['float64', 'int64'])
df_4num.info()
#sns_plot= sns.pairplot(df_4num.drop(["CAUSE_OF_DEATH","UNLISTED_REASON" ],axis=1), hue="Drop_3M", size=3)

#sns_plot.savefig("output.png")
