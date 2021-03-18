#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix,plot_roc_curve
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.model_selection import KFold,cross_val_score,train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE
SMOTE=SMOTE(random_state=0)


# In[2]:


##Build KNN model for the Excellent group
remove=[0,16,-1]
df=pd.read_csv("FICO_excellent.csv")
df.drop(df.columns[remove],axis=1,inplace=True)
y=df["status_time"].values
newdf = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
df=newdf.round(3)
x=df.drop("status_time", axis=1).values
train_x1, test_x1, train_y1, test_y1=train_test_split(x, y, train_size=0.8,random_state=0)
print("default before SMOTE:", len(train_y1[train_y1==1]))
print("pay off before SMOTE:", len(train_y1[train_y1==2]))
train_x1, train_y1=SMOTE.fit_sample(train_x1, train_y1)
print("default after SMOTE:", len(train_y1[train_y1==1]))
print("pay off after SMOTE:", len(train_y1[train_y1==2]))
model1=KNeighborsClassifier(13)
model1.fit(train_x1, train_y1)


# In[3]:


#Build KNN model for the good group
df=pd.read_csv("FICO_good.csv")
df.drop(df.columns[remove],axis=1,inplace=True)
y=df["status_time"].values
newdf = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
df=newdf.round(3)
x=df.drop("status_time", axis=1).values
train_x2, test_x2, train_y2, test_y2=train_test_split(x, y, train_size=0.8,random_state=0)
print("default before SMOTE:", len(train_y2[train_y2==1]))
print("pay off before SMOTE:", len(train_y2[train_y2==2]))
train_x2, train_y2=SMOTE.fit_sample(train_x2, train_y2)
print("default after SMOTE:", len(train_y2[train_y2==1]))
print("pay off after SMOTE:", len(train_y2[train_y2==2]))
model2=KNeighborsClassifier(13)
model2.fit(train_x2, train_y2)


# In[4]:


#Build KNN model for the low group
df=pd.read_csv("FICO_low.csv")
df.drop(df.columns[remove],axis=1,inplace=True)
y=df["status_time"].values
newdf = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
df=newdf.round(3)
x=df.drop("status_time", axis=1).values
train_x3, test_x3, train_y3, test_y3=train_test_split(x, y, train_size=0.8,random_state=0)
print("default before SMOTE:", len(train_y3[train_y3==1]))
print("pay off before SMOTE:", len(train_y3[train_y3==2]))
train_x3, train_y3=SMOTE.fit_sample(train_x3, train_y3)
print("default after SMOTE:", len(train_y3[train_y3==1]))
print("pay off after SMOTE:", len(train_y3[train_y3==2]))
model3=KNeighborsClassifier(13)
model3.fit(train_x3, train_y3)


# In[5]:


#Find the confusion matrix of three groups
y_pred1 = model1.predict_proba(test_x1)
y_pred10 = model1.predict(test_x1)
print(confusion_matrix(test_y1-1, y_pred10-1))
print(classification_report(test_y1-1, y_pred10-1))
fpr1, tpr1, threshold1 = roc_curve(test_y1,y_pred1[:,-1],pos_label=2)
roc_auc1 = auc(fpr1, tpr1)
y_pred2 = model2.predict_proba(test_x2)
y_pred20 = model2.predict(test_x2)
print(confusion_matrix(test_y2-1, y_pred20-1))
print(classification_report(test_y2-1, y_pred20-1))
fpr2, tpr2, threshold2 = roc_curve(test_y2,y_pred2[:,-1],pos_label=2)
roc_auc2 = auc(fpr2, tpr2)
y_pred3 = model3.predict_proba(test_x3)
y_pred30 = model2.predict(test_x3)
print(confusion_matrix(test_y3-1, y_pred30-1))
print(classification_report(test_y3-1, y_pred30-1))
fpr3, tpr3, threshold3 = roc_curve(test_y3,y_pred3[:,-1],pos_label=2)
roc_auc3 = auc(fpr3, tpr3)


# In[6]:


#Draw the ROC plot comparision
plt.figure(figsize=(10,10))
plt.plot(fpr1, tpr1, label = 'Excellent group'+', AUC = %0.3f' % roc_auc1,alpha=0.9)
plt.plot(fpr2, tpr2, label = 'Good group'+', AUC = %0.3f' % roc_auc2,alpha=0.9)
plt.plot(fpr3, tpr3, label = 'Low group'+', AUC = %0.3f' % roc_auc3,alpha=0.9)
plt.plot([0,1],[0,1],"r--")
plt.legend(loc = 'lower right')
plt.title("ROC of KNN model with FICO segmentations",fontsize=15)
plt.ylabel('True Positive Rate',fontsize=12)
plt.xlabel('False Positive Rate',fontsize=12)
plt.show()


# In[7]:


#Visualize the confusion matrix
cm1=confusion_matrix(test_y1-1, y_pred10-1)
cm2=confusion_matrix(test_y2-1, y_pred20-1)
cm3=confusion_matrix(test_y3-1, y_pred30-1)
plt.matshow(cm1,cmap=plt.cm.Blues)
plt.colorbar()
plt.annotate(cm1[0,1],xy=(0,1))
plt.annotate(cm1[1,1],xy=(1,1))
plt.annotate(cm1[0,0],xy=(0,0))
plt.annotate(cm1[1,0],xy=(1,0))
plt.ylabel("Predicted Values")
plt.xlabel("Actual Values")
plt.title("Excellent Group")
plt.show()


# In[8]:


plt.matshow(cm2,cmap=plt.cm.Greens)
plt.colorbar()
plt.annotate(cm2[0,1],xy=(0,1))
plt.annotate(cm2[1,1],xy=(1,1))
plt.annotate(cm2[0,0],xy=(0,0))
plt.annotate(cm2[1,0],xy=(1,0))
plt.ylabel("Predicted Values")
plt.xlabel("Actual Values")
plt.title("Good Group")
plt.show()


# In[9]:


plt.matshow(cm3,cmap=plt.cm.Reds)
plt.colorbar()
plt.annotate(cm3[0,1],xy=(0,1))
plt.annotate(cm3[1,1],xy=(1,1))
plt.annotate(cm3[0,0],xy=(0,0))
plt.annotate(cm3[1,0],xy=(1,0))
plt.ylabel("Predicted Values")
plt.xlabel("Actual Values")
plt.title("Low Group")
plt.show()

