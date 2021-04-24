#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("dawn.csv")


# In[3]:


df.drop("id",axis=1,inplace=True)
y=df["status_time"].values
x=df.drop("status_time", axis=1).values


# In[4]:


x


# In[5]:


df


# In[6]:


y


# In[7]:


kf=KFold(n_splits=5,random_state=2,shuffle=True)


# In[8]:


train_x, test_x, train_y, test_y=train_test_split(x, y, train_size=0.8)


# In[ ]:


model = svm.SVC(kernel='poly', gamma=1) 
model.fit(train_x, train_y)


# In[ ]:


model.score(test_x, test_y)


# In[ ]:




