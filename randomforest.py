# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:41:31 2015

@author: Neo
"""

import numpy as np
import pandas as pd

#---------------------------#
#-----training data---------#
#---------------------------#

#---read in training data---#
df = pd.read_csv('E:/PrWS/Kaggle/Titanic/train.csv', header=0)
#---data munging---#
#create a new column called Gender with female being 0, male being 1
df['Gender'] = None
df['Gender'] = df['Sex'].map({"female":0,"male":1}).astype(int)

#fill in missing values for Age
median_ages = np.zeros((2,3))
for i in range(2): # i for gender
    for j in range(3): # j for Pclass
        median_ages[i,j] = df[(df.Gender == i) & (df.Pclass == (j+1))]['Age'].dropna().median()

df['AgeFill'] = df['Age']
for i in range(2):
    for j in range(3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == (j+1)),'AgeFill']= median_ages[i,j]

#convert Embarked to numeric values, new column 'Em'
#S:0,C:1,Q:2, nan to be removed
pd.Series(df.Embarked.ravel()).unique()
df['Em'] = None
#df[df.Embarked.isnull()]['Embarked'] = 0 will fail
df.loc[df.Embarked.isnull(),'Embarked'] = 'S'
df.Em = df.Embarked.map({'S':0,'C':1,'Q':2}).astype(int)

#drop columns
df = df.drop(['PassengerId','Name','Sex','Age','Ticket','Cabin'],axis = 1)

#drop nan
df = df.dropna()

df.info()
df.describe()

#convert to Numpy array
train_data = df.values


#---------------------------#
#------testing data---------#
#---------------------------#
dft = pd.read_csv('E:/PrWs/Kaggle/Titanic/test.csv',header = 0)
dft['Gender'] = None
dft['Gender'] = dft['Sex'].map({"female":0,"male":1}).astype(int)
dft['AgeFill'] = dft['Age']
#use the statistic from the training set, since it has more observations
for i in range(2):
    for j in range(3):
        dft.loc[(dft.Age.isnull()) & (dft.Gender == i) & (dft.Pclass == (j + 1)),'AgeFill'] = median_ages[i,j]
    
dft.loc[dft.Fare.isnull(), 'Fare'] = dft.Fare.mean()
dft['Em'] = None
dft['Em'] = dft['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
dft = dft.drop(['PassengerId','Age','Name','Sex','Ticket','Cabin','Embarked'],axis = 1)
test_data = dft.values


#---------------------------#
#-----------model-----------#
#---------------------------#

#Using the predictive capabilities of the scikit-learn package is very simple. 
#In fact, it can be carried out in three simple steps: initializing the model, 
#fitting it to the training data, and predicting new values.


from sklearn.ensemble import RandomForestClassifier
# Create the random forest object including all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
output = forest.predict(test_data).astype(int)


#---------------------------#
#--------write result-------#
#---------------------------#
import csv
prediction_file = open("SubmissionRF.csv",'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId","Survived"])
for id, pred in enumerate(output):
    prediction_file_object.writerow([id+892, pred])

prediction_file.close()



