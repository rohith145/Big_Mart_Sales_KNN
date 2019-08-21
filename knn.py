# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 00:16:37 2019

@author: rogunda
"""
#why I imported the above libraries can be understood by seeing attached Jupyter Notebook
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 
from math import sqrt


#This will help us clean data "Item_Weight"has many missing values fill it with median and "Outlet_Size" is a categorical value fill it with mode
def clean_data(df):
    df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
    mode = df['Outlet_Size'].mode()
    df['Outlet_Size'].fillna(mode[0], inplace =True)
    df.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)#dropping unnecessary columns
    df = pd.get_dummies(df)#dealing with categorical variables
    return df

def scaling(df): 
    sc_x=StandardScaler()
    df=sc_x.fit_transform(df)
    return df
'''
def estimater(model,x_train,y_train,x_test):
    model.fit(x_train, y_train)  #fit the model
    y_pred=model.predict(x_test) #make prediction on test set
    return y_pred
'''
def grid_search(x_train,y_train,x_test):
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5) 
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    return y_pred

df_train=pd.read_csv("knn_train1.csv")
df_test=pd.read_csv("knn_test11.csv")
train_dummy=clean_data(df_train)
test_dummy=clean_data(df_test)
train_dummy = train_dummy.drop('Item_Outlet_Sales', axis=1)
train_dummy=scaling(train_dummy)
test_dummy=scaling(test_dummy)
y_train=df_train['Item_Outlet_Sales']
y_pred=grid_search(train_dummy,y_train,test_dummy)
output_knn=pd.read_csv("output_knn.csv")
output_knn['Item_Outlet_Sales']=y_pred
output_knn.to_csv('output_knn1.csv')