# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier 

# IMPORT DATA
df=pd.read_csv('C:/Users/Josko/Desktop/my_projects/Mach_L/NN/neural_network/squared_sc/squared_3.csv')
df=df.sample(frac=1)
print(df.head())

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

# SPLIT THE DATA TRAINING/TESTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# KN CLASSIFIER
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('KN CLASSIFIER')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# SVM CLASSIFIER
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('SVM CLASSIFIER')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# DECISION TREE CLASSIFIER
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('DECISION TREE CLASSIFIER')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# RANDOM FOREST CLASSIFIER
classifier = RandomForestClassifier() # n_estimators 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('RANDOM FOREST CLASSIFIER')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
