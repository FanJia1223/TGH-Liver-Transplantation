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

# Drop unnecessary columns 
df = pd.read_excel('Cleaned Master File 2018-12-17.xlsx')
df_val = df
df_val.drop(['BW_1mo','BW_2mo','BW_3mo','BW_4mo','BW_5mo','BW_6mo','BW_7mo','BW_8mo','BW_9mo',
             'BW_10mo','BW_11mo','BW_1yr','BW_>1yr','Unlisted_Lab_Int','Identifier','Identifier.1','Identifier'], inplace=True, axis=1, errors='ignore')
df_val.drop(['SUBJECT_ID.2','LISTING_DATE.2','DATE_OF_LAB.1','Identifier','BW At Listing','SUBJECT_ID.1','LISTING_DATE.1'],inplace=True, axis=1, errors='ignore')
df_val.drop(['DIALYSIS_TYPE_DESCR'], inplace=True, axis=1, errors='ignore')
df_val.drop(['DATE_OF_TRANSPLANT','AGE_AT_UNLISTING','AGE_AT_DEATH','DATE_OF_LAB','LAB_INTERVAL',
             'Identifier','LAB_INTERVAL.1'], inplace=True, axis=1, errors='ignore')
df_val.drop(['EST_GFR'], inplace=True, axis=1, errors='ignore')
patients = df_val.dropna(how='any')
patients.info()

# quick look at the data 
table1 = np.mean(patients, axis = 0)
table2 = np.std(patients, axis = 0 )

feature_patients = ['CREAT',	'BILT',	'AST',	'ALT',	'ALK_PHOS',	'PLATELETS',	'WBC',	'ALB',	'MELD',	'HGB',	'INR',	'AGE_AT_LISTING',	'PATIENT_STATUS',	'TREATMENT_PHASE',	'SEX',	'RACE',	'HEIGHT_AT_LISTING',	'WEIGHT_AT_LISTING',	'BLOOD_TYPE',	'RH_FACTOR',	'P_DIAG',	'SEC_DIAG',	'SMOKER',	'PREV_SURG',	'DIALYSIS_TYPE',	'Na',	'HCO3',	'GLUCOSE',	'CALC',	'MG',	'PHOS','LISTING_TIME']

## Data Exploration
patients['Drop_3M'].value_counts()

sns.countplot (x = 'Drop_3M', data= patients, palette = 'hls')
plt.show()
plt.savefig('count_plot.png')

X_patients = patients[feature_patients]
y_patients = patients['Drop_3M']
target_states = ['Drop_off', 'Live']

###Visualization 

table = pd.crosstab(patients['Na'] ,patients['Drop_3M'])
table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
plt.title('State of Patient for Na')
plt.xlabel('Na')
plt.ylabel('State')
plt.savefig('Na_states.png')


#Spliting the training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X_patients, y_patients, random_state=0)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=5,scoring='accuracy')
#need to look at scoring criteria 

clf =logreg_cv.fit(X_train,y_train)
logreg_cv.score(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

#implementing the model 
#import statsmodels.api as sm 
#logit_model =  sm.Logit(y_train, X_train)
#result = logit_model.fit()
#print(result.summary2())


#Predicting the test set results and calculating the acurracy
y_pred = logreg_cv.predict(X_test)
print ('Accuracy of logisitic regression classifier on the test set:{:.2f}'.format
       (logreg_cv.score(X_test, y_test)))

# Building a confusion matrix 

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logreg_cv.predict(X_test),y_test)
print(confusion_matrix)


##compute precisio, recall, F-measure and support 
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#logit1.fit(inputData,outputData)
clf.score(X_test, y_test)
##Computing false and true positive rates
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, X_patients, y_patients, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

fpr, tpr,_ = roc_curve( y_test,y_pred, drop_intermediate=False)
#clf.predict(X_test)




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

AUC = roc_auc_score( y_test,clf.predict(X_test))
print('AUC is :', AUC)
#



#plt.figure(0).clf()
#plt.title('ROC')
#y_true = np.array(y_test)
#y_pred = np.array(clf.predict(X_test))
#fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
#auc = metrics.roc_auc_score(y_true, y_pred)
#plt.plot(fpr,tpr,label="Logistic Regression, AUC=%0.2f" % auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


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
