#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Reference 1: Codes and hints provided by Prof. Matthew D Murphy

# Importing necessary libraries

import sklearn
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 

from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score


# In[9]:


pd.options.display.max_columns = 2000


# In[2]:


# loading the credit card default dataset

cc_df = pd.read_csv("ccdefault.csv")


# In[13]:


# droppinng "ID" column from the dataset

cc_df = cc_df.drop(["ID"],axis =1)


# In[19]:


# Top 5 samples of the dataset

cc_df.head()


# In[15]:


# Summary Statistics of the dataset

cc_df.describe()


# In[16]:


# Checking for missing values

cc_df.isnull().describe()


# In[44]:


# Creating feature variables and target variable

X = cc_df.drop(["DEFAULT"], axis =1)
y = cc_df["DEFAULT"]


# In[45]:


# Checking dimension of X and y

print(X.shape,"\n",y.shape)


# In[46]:


# printing top 5 samples of X and y

print(X.head(),"\n",y.head())


# In[47]:


# Checking types of all the variables

cc_df.dtypes


# In[48]:


from sklearn.preprocessing   import StandardScaler


# In[49]:


# normalisation of data

sc_x  = StandardScaler()

X = sc_x.fit_transform(X)


# In[50]:


# printing samples after normalisation

print(X.shape,"\n",y.shape)

print(X[0:5],"\n",y[0:5])


# In[76]:


import time


# In[77]:


# timing and runnning the model with 10 different random states

s_rs_t = time.time()

accuracy_scores_train = []
accuracy_scores_test  = []

for i in range(1,11):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1, stratify = y, random_state = i)
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    
    y_train_pred = dt.predict(X_train)
    y_test_pred  = dt.predict(X_test)
    
    accuracy_scores_train.append(accuracy_score(y_train,y_train_pred))
    accuracy_scores_test.append(accuracy_score(y_test,y_test_pred))
    
print("accuracy scores training set: ", accuracy_scores_train,"\n")
print("accuracy scores test set: ", accuracy_scores_test,"\n\n")


mean_acc_score_train   = np.mean(accuracy_scores_train)
mean_acc_score_test    = np.mean(accuracy_scores_test)

stdev_acc_scores_train = np.std(accuracy_scores_train)
stdev_acc_scores_test  = np.std(accuracy_scores_test)

print ("mean of accuracy scores on training set: ", mean_acc_score_train,"\n")
print ("mean of accuracy scores on test set: ",mean_acc_score_test, "\n" )

print ("stdev of accuracy scores on training set: ", stdev_acc_scores_train, "\n")
print ("stdev of accuracy scores on test set: ", stdev_acc_scores_test, "\n" )

e_rs_t = time.time()

print("time taken for samples with multiple random states: ",(e_rs_t-s_rs_t))
    


# In[83]:


# timing and running the model using 10 fold cross-validation

from sklearn.model_selection import cross_val_score

s_kf_t = time.time()

dtc = DecisionTreeClassifier()

scores = cross_val_score(dtc, X_train,y_train,
                         cv = 10)

print("CV accuracy scores: ",scores,"\n")

print("CV Accuracy mean: ", np.mean(scores),"\n")
print("CV Accuracy stdev: ", np.std(scores),"\n")

e_kf_t = time.time()

print("time taken for k-fold cross validation model selection: ",(e_kf_t-s_kf_t),"\n")


# In[81]:


print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is: rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




