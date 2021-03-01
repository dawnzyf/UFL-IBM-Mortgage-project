#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, classification_report, plot_roc_curve, precision_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier


# In[2]:


#load data
data=pd.read_csv("mortgage_final_data.csv")


# In[3]:


#define independent and dependent variables
y=data["status_time"].values
x=data.drop(["status_time","id"], axis=1).values


# In[4]:


#print distribution of dependent variable
print("default:", len(data[data["status_time"]==1]))
print("pay off:", len(data[data["status_time"]==2]))


# In[5]:


#split train and test set
train_x, test_x, train_y, test_y=train_test_split(x, y, train_size=0.8,random_state=7)


# In[6]:


#display the number of train set
print("default before SMOTE:", len(train_y[train_y==1]))
print("pay off before SMOTE:", len(train_y[train_y==2]))


# In[7]:


#use SMOTE to deal with imbalanced data
SM=SMOTE(random_state=0)
S_train_x, S_train_y=SM.fit_sample(train_x, train_y)


# In[8]:


#display the number of training set after SMOTE
print("default after SMOTE:", len(S_train_y[S_train_y==1]))
print("pay off after SMOTE:", len(S_train_y[S_train_y==2]))


# In[9]:


#Train the model
model=LGBMClassifier(n_estimators=100,metric="auc")
model.fit(S_train_x,S_train_y,eval_set=[(test_x, test_y)])


# In[10]:


#Calculate the confusion matrix
y_pred = model.predict(test_x)
cm=confusion_matrix(test_y,y_pred)
print(classification_report(test_y,y_pred))


# In[11]:


#Visualize the confusion matrix
plt.matshow(cm,cmap=plt.cm.Greens)
plt.colorbar()
plt.annotate(cm[0,1],xy=(0,1))
plt.annotate(cm[1,1],xy=(1,1))
plt.annotate(cm[0,0],xy=(0,0))
plt.annotate(cm[1,0],xy=(1,0))
plt.ylabel("Predicted Values")
plt.xlabel("Actual Values")
plt.title("Confusion Matrix for LGBM model")
plt.show()


# In[12]:


#Find the average accuracy of K-Fold Validation(K=5)
cross_val_score(model, S_train_x, S_train_y, cv=5, scoring="accuracy").mean()


# In[13]:


#Draw the ROC curve and calculate the AUC.
fig, ax=plt.subplots(figsize=(8,8))
plot_roc_curve(model, test_x, test_y, ax=ax, label="LGBM, AUC=0.868",color="lightcoral")
plt.title("ROC curve of LGBM model",fontsize=15)
plt.plot([0,1],[0,1],"g--")
plt.show()


# In[ ]:




