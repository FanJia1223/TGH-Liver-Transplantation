# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)

patients = pd.read_excel('testing.xlsx')

feature_patients = ['ALB', 'BILT', 'CREAT', 'INR', 'Na', 'Dialysis', 'ListYears', 'Age', 'MELD']

X_patients = patients[feature_patients]
y_patients = patients['Mortality']
target_states = ['Death', 'Live']

X_train, X_test, y_train, y_test = train_test_split(X_patients, y_patients, random_state=0)

from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import (
plot_class_regions_for_classifier_subplot)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

clf = LogisticRegression(C=100).fit(X_train, y_train)

plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
                                         None, 'Logistic regression for binary classification\nBloodwork dataset: Death vs Live',
                                         subaxes)
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
## we must apply the scaling to the test set that we computed for the training set
#X_test_scaled = scaler.transform(X_test)
#
#knn = KNeighborsClassifier(n_neighbors = 5)
#knn.fit(X_train_scaled, y_train)
#print('Accuracy of K-NN classifier on training set: {:.2f}'
#     .format(knn.score(X_train_scaled, y_train)))
#print('Accuracy of K-NN classifier on test set: {:.2f}'
#     .format(knn.score(X_test_scaled, y_test)))

example_patient = [[31, 36, 96, 1.35, 133, 0, 0.47, 58.48, 13]]

print(clf.predict([example_patient])[0])

#example_patient_scaled = scaler.transform(example_patients)
#print('Predicted patients state for ', example_patient, ' is ', 
#          target_states[knn.predict(example_patients_scaled)[0]-1])
