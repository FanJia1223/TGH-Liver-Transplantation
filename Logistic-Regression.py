# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set (style = 'white')
sns.set (style = 'whitegrid', color_codes = True)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)

patients = pd.read_excel('fake-testing-data.xlsx')

# quick look at the data 
table1 = np.mean(patients, axis = 0)
table2 = np.std(patients, axis = 0 )

feature_patients = ['Na',	'GLUCOSE',	'AGE_AT_LISTING',	'TREATMENT_PHASE',	'TX_NUMBER',	'SEX',	'RACE',	'HEIGHT_AT_LISTING',	'WEIGHT_AT_LISTING',	'BLOOD_TYPE',	'RH_FACTOR',	'P_DIAG',	'SEC_DIAG',	'SMOKER',	'DIALYSIS_TYPE',	'CREAT',	'BILT',	'AST',	'ALT',	'ALK_PHOS',	'PLATELETS',	'WBC',	'ALB',	'EST_GFR',	'MELD',	'HGB',	'INR',	'listing time ']

## Data Exploration
patients['Final State'].value_counts()

sns.countplot (x = 'Final State', data= patients, palette = 'hls')
plt.show()
plt.savefig('count_plot.png')

X_patients = patients[feature_patients]
y_patients = patients['Final State']
target_states = ['Death', 'Live']

###Visualization 

table = pd.crosstab(patients['Na'] ,patients['Final State'])
table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.title('State of Patient for Na')
plt.xlabel('Na')
plt.ylabel('State')
plt.savefig('Na_states.png')



X_train, X_test, y_train, y_test = train_test_split(X_patients, y_patients, random_state=0)

from sklearn.linear_model import LogisticRegression


logit1 = LogisticRegression(C=5)
clf =logit1.fit(X_train, y_train)
logit1.score(X_train, y_train)


#implementing the model 
#import statsmodels.api as sm 
#logit_model =  sm.Logit(y_train, X_train)
#result = logit_model.fit()
#print(result.summary2())


#Predicting the test set results and calculating the acurracy
y_pred = logit1.predict(X_test)
print ('Accuracy of logisitic regression classifier on the test set:{:.2f}'.format
       (logit1.score(X_test, y_test)))

# Building a confusion matrix 

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(X_train),y_train)
print(confusion_matrix)


##compute precisio, recall, F-measure and support 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#logit1.fit(inputData,outputData)
clf.score(X_test, y_test)
##Computing false and true positive rates
fpr, tpr,_ = roc_curve(clf.predict(X_test), y_test, drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

AUC = roc_auc_score(clf.predict(X_test), y_test)
print('AUC is :', AUC)

##Model Coefficients 


##Visualising boundaries 
#plt.figure()
#plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=logit1.predict_proba(inputData)[:,1],alpha=0.4)
#plt.xlabel('Glucose level ')
#plt.ylabel('BMI ')
#plt.show()
#
#plt.figure()
#plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=outputData,alpha=0.4)
#plt.xlabel('Glucose level ')
#plt.ylabel('BMI ')
#plt.show()
