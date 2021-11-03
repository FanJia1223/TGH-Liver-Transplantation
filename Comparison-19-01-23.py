#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:09:20 2019

@author: jiafan
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns 
import graphviz 
from sklearn import tree
from sklearn.model_selection import train_test_split

sns.set (style = 'white')
sns.set (style = 'whitegrid', color_codes = True)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)

# Drop unnecessary columns 
df = pd.read_excel('Testing-Data-Jan-24.xlsx')

df_val = df[:]
df_val.drop(['BW_1mo','BW_2mo','BW_3mo','BW_4mo','BW_5mo','BW_6mo','BW_7mo','BW_8mo','BW_9mo',
             'BW_10mo','BW_11mo','BW_1yr','BW_>1yr','Unlisted_Lab_Int','Identifier','Identifier.1','Identifier'], inplace=True, axis=1, errors='ignore')
df_val.drop(['SUBJECT_ID.2','LISTING_DATE.2','DATE_OF_LAB.1','Identifier','BW At Listing','SUBJECT_ID.1','LISTING_DATE.1'],inplace=True, axis=1, errors='ignore')
df_val.drop(['DIALYSIS_TYPE_DESCZR'], inplace=True, axis=1, errors='ignore')
df_val.drop(['DATE_OF_TRANSPLANT','AGE_AT_UNLISTING','AGE_AT_DEATH','DATE_OF_LAB','LAB_INTERVAL',
             'Identifier','LAB_INTERVAL.1','DATE_OF_TRANSPLANT.1','AGE_AT_DEATH.1'], inplace=True, axis=1, errors='ignore')
df_val.drop(['EST_GFR'], inplace=True, axis=1, errors='ignore')
df_val.count()

patients = df_val.dropna(how='any')
patients.count()


# quick look at the data 
#table1 = np.mean(patients, axis = 0)
#table2 = np.std(patients, axis = 0 )

feature_patients = ['CREAT','BILT','AST',	'ALT','ALK_PHOS','PLATELETS','WBC',	'ALB',	'MELD',	'HGB',	'INR',	'AGE_AT_LISTING',	'PATIENT_STATUS',	'TREATMENT_PHASE',	'SEX',	'RACE',	'HEIGHT_AT_LISTING',	'WEIGHT_AT_LISTING',	'BLOOD_TYPE',	'RH_FACTOR',	'P_DIAG',	'SEC_DIAG',	'SMOKER',	'PREV_SURG',	'DIALYSIS_TYPE',	'Na',	'HCO3',	'GLUCOSE',	'CALC',	'MG',	'PHOS','LISTING_TIME']

## Data Exploration
patients['Drop_3M'].value_counts()

#sns.countplot (x = 'Drop_3M', data= patients, palette = 'hls')
#plt.show()
#plt.savefig('count_plot.png')

X_patients = patients[feature_patients]
y_patients = patients['Drop_3M']
target_states = ['Drop_off', 'Live']

X_train, X_test, y_train, y_test = train_test_split(X_patients, y_patients,test_size = 0.3 ,random_state=5)

# Building the logistic regression model 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-5,5,11), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=5,scoring='accuracy')
#need to look at scoring criteria 

clf =logreg_cv.fit(X_train,y_train)
clf.score(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",clf.best_params_)
print("accuracy :",clf.best_score_)

y_pred = clf.predict_proba(X_test)
y_pred_rounded = clf.predict(X_test)
print ('Accuracy of logisitic regression classifier on the test set:{:.5f}'.format
       (clf.score(X_test, y_test)))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#logit1.fit(inputData,outputData)
##Computing false and true positive rates
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, X_patients, y_patients, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

fpr, tpr, threshold = roc_curve( y_test,y_pred[:,1], pos_label = 1)
#auc = roc_auc_score( y_test,clf.predict(X_test))
auc = roc_auc_score( y_test,y_pred[:,1])

# Bulding the decision tree model 
sample_split_range = list(range(2, 50))
param_grid = dict(min_samples_split=sample_split_range)

# instantiate the grid
dtc = tree.DecisionTreeClassifier(max_depth = 5)
dtc_cv = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
clf_dct=dtc_cv.fit(X_train, y_train)
#clf_dct1 = tree.DecisionTreeClassifier(max_depth = 5).fit(X_train,y_train)
clf_dct1  = dtc_cv.best_estimator_
dot_data = tree.export_graphviz(clf_dct1, out_file=None, 
                     feature_names=feature_patients,  
                     class_names=target_states,  
                     filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.format = 'png'
graph.render('dtree_render',view=True)

#clf = dtc_cv.best_estimator_
y_pred_dct = dtc_cv.predict_proba(X_test)
y_pred_dct_rounded = dtc_cv.predict(X_test)

print(clf_dct.best_score_)

# Dictionary containing the parameters (min_samples_split) used to generate that score
print(clf_dct.best_params_)

# Actual model object fit with those best parameters
# Shows default parameters that we did not specify
print(clf_dct.best_estimator_)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#logit1.fit(inputData,outputData)
clf_dct.score(X_test, y_test)
##Computing false and true positive rates
fpr_dct, tpr_dct, thresholds_dct = roc_curve( y_test,y_pred_dct[:,1], drop_intermediate=False)
#clf_dct.predict(X_test)
#auc_dct = roc_auc_score( y_test,clf_dct.predict(X_test))
auc_dct = roc_auc_score( y_test,y_pred_dct[:,1])
#clf_dct1 = tree.DecisionTreeClassifier(max_depth = 5).fit(X_train,y_train)
#dot_data = tree.export_graphviz(clf_dct1, out_file=None, 
#                     feature_names=feature_patients,  
#                     class_names=target_states,  
#                     filled=True, rounded=True,  
#                      special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph.format = 'png'
#graph.render('dtree_render',view=True)


# Draw Comparison Graph
plt.figure(0)
plt.title('ROC')
plt.plot(fpr,tpr,label="LR, AUC=%0.2f" % AUC)
plt.plot(fpr_dct,tpr_dct,label="DT, AUC_dct=%0.2f" % AUC_dct)
plt.plot(fpr_meld,tpr_meld,label="MELD, AUC=%0.2f" % auc_meld)
plt.plot(fpr_na,tpr_na,label="MELD-Na, AUC=%0.2f" % auc_na)
plt.plot(fpr_OPOM,tpr_OPOM,label="OPOM, AUC=%0.2f" % auc_OPOM)
#
#fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=1)
#auc = metrics.roc_auc_score(y_true, scores)
#plt.plot(fpr,tpr,label="OPOM, AUC=%0.2f" % auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
