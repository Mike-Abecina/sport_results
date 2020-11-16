#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import pyodbc
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from pandas import DataFrame


from sklearn.model_selection import train_test_split
import tensorflow as tf


data = pd.read_csv(r"C:\Users\maa\OneDrive - William Clarke College\documents\Physical Activity Survey Results.csv")


data = data[['ID', 'Start time', 'Completion time', 'What\'s your gender?', 'How old are you?',
       'How many hours of exercise do you do per week?',
       'What is the average intensity of your exercise?',
       'What type of exercise do you do?',
       'Is your exercise voluntary or compulsory?',
       'Do you exercise in a group or by yourself?',
       'How accessible are sporting facilities to you?']]


def sport_cardio(row):
    if 'Cardio' in row['What type of exercise do you do?']:
        return 1
    else:
        return 0
    
def sport_strength(row):
    if 'Strength' in row['What type of exercise do you do?']:
        return 1
    else:
        return 0

def sport_competitive(row):
    if 'Competitive' in row['What type of exercise do you do?']:
        return 1
    else:
        return 0
    
def sport_recreational(row):
    if 'Recreational' in row['What type of exercise do you do?']:
        return 1
    else:
        return 0
    
def time_convert(row):
    if row['How many hours of exercise do you do per week?'] == '1-2 hours':
        return 1.5
    if row['How many hours of exercise do you do per week?'] == '3-4 hours':
        return 3.5
    if row['How many hours of exercise do you do per week?'] == 'Less than 1 hour':
        return 0.5
    if row['How many hours of exercise do you do per week?'] == 'More than 5 hours':
        return 7.5
    
def age_convert(row):
    if row['How old are you?'] == '11-15':
        return 13
    if row['How old are you?'] == '16-20':
        return 16
    if row['How old are you?'] == '21-30':
        return 25
    if row['How old are you?'] == '31-40':
        return 35
    if row['How old are you?'] == '41-50':
        return 45
    if row['How old are you?'] == '50+':
        return 55
    else:
        return 7.5
    
def gender_num(row):
    if row['What\'s your gender?']=='Male':
        return 1
    if row['What\'s your gender?']=='Female':
        return 0
    else:
        return 2


data['cardio']=data.apply(lambda x:sport_cardio(x), axis = 1)
data['strength']=data.apply(lambda x:sport_strength(x), axis = 1)
data['competitive']=data.apply(lambda x:sport_competitive(x), axis = 1)
data['recreational']=data.apply(lambda x:sport_recreational(x), axis = 1)
data['time'] = data.apply(lambda x: time_convert(x), axis =1)
data['age'] = data.apply(lambda x: age_convert(x), axis = 1 )
data['gender'] = data.apply(lambda x:gender_num(x), axis =1)



data = pd.concat([data, pd.get_dummies(data['Is your exercise voluntary or compulsory?'])], axis =1)
data = pd.concat([data, pd.get_dummies(data['Do you exercise in a group or by yourself?'])], axis =1)


data = data[['What is the average intensity of your exercise?',
       'How accessible are sporting facilities to you?', 'cardio', 'strength',
       'competitive', 'recreational', 'time', 'age', 'Compulsory;','gender',
       'Compulsory;Voluntary ;', 'Voluntary ;', 'Voluntary ;Compulsory;',
       'By myself;', 'By myself;In a group;', 'In a group;',
       'In a group;By myself;']]



data = data.astype(int)

data.rename(columns =  {'Compulsory;':'Compulsory','What is the average intensity of your exercise?':'What_is_the_average_intensity_of_your_exercise'
       ,'How accessible are sporting facilities to you?':'How_accessible_are_sporting_facilities_to_you','Compulsory;Voluntary ;':'Compulsory_Voluntary', 'Voluntary ;':'Voluntary', 'Voluntary ;Compulsory;':'Voluntary_Compulsory',
       'By myself;': 'By_Myself', 'By myself;In a group;':'By_myself_in_a_group', 'In a group;':'In_a_group',
       'In a group;By myself;':'In_a_group_By_myself'}, inplace = True)


data.rename(columns = {'gender':'Target'}, inplace = True)

data = data[data['Target']!=2]
data['Target']= data['Target'].astype('bool')
data.dtypes

X_train, X_test= train_test_split(data, test_size=0.35)

X_train.columns
Y_train =X_train.pop('Target')
Y_test = X_test.pop('Target')


def make_input_fn(data_df, label_df, num_epochs=100, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(X_train, Y_train)
eval_input_fn = make_input_fn(X_test, Y_test, num_epochs=1, shuffle=False)


features = list(X_train.columns)
feature_columns = [tf.feature_column.numeric_column(k) for k in features]




feature_columns


# In[31]:


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)


print(result)



