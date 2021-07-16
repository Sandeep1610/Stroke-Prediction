#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd                                             #importing pandas and numpy library
import numpy as np
from sklearn.model_selection import train_test_split             # to check our model accuracy
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_excel(r'C:\Users\hp\Desktop\DataSets\healthcare-dataset-stroke-data.xlsx')          # path of Excel file


# In[ ]:





# In[3]:


print(data.shape)                 # 5 feature columns and 1 label column
data.head()                       # 1000 training examples


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


avg=data['bmi'].mean()


# In[7]:


avg


# In[8]:


data.bmi=(data.bmi.fillna(avg))


# In[9]:


data.isnull().sum()


# In[10]:


data.describe()


# In[11]:


data['work_type'] = data['work_type'].map({'Private':0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4})


# In[12]:


data['gender'] = data['gender'].map({'Male':0, 'Female':1})
data['Residence_type'] = data['Residence_type'].map({'Urban':0, 'Rural':1})
data['smoking_status'] = data['smoking_status'].map({'formerly smoked':0, 'never smoked':1, 'smokes':2, 'Unknown':3})
data['ever_married'] = data['ever_married'].map({'Yes':0, 'No':1})


# In[13]:


data.head()


# In[14]:


features = ['id','age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'Residence_type',
 'avg_glucose_level',
 'bmi',
 'gender',
 'work_type',
 'smoking_status']
label=['stroke']
X = data[features]
Y = data[label]


# In[15]:


X.isnull().sum()


# In[16]:


X.gender=(X.gender.fillna(1))


# In[17]:


X.isnull().sum()


# In[18]:


X = X.drop(columns=['id'])


# In[19]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X = sc.fit_transform(X)   # x=(x-mean)/sd


# In[ ]:





# In[20]:


from imblearn.over_sampling import SMOTE           #Synthetic Minority Oversampling Technique

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(X, Y)    


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x_smote, y_smote,test_size=0.25,random_state=100)


# In[22]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# # using LogisticRegression model

# In[25]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)


# In[26]:


y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg
print(np.shape(X_test))


# In[27]:


from sklearn import metrics                              # to check our accuracy,precision and recall by using our test set
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred_log_reg)
cnf_matrix


# In[28]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_log_reg))
print("Precision:",metrics.precision_score(Y_test, y_pred_log_reg))
print("Recall:",metrics.recall_score(Y_test, y_pred_log_reg))


# In[29]:


log_reg.score(X_test,Y_test)


# # Using KNeighborsClassifier model

# In[30]:


#KNN
#training model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p = 2)
knn.fit(X_train,Y_train)

#getting confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
print('confusion matrix:\n',cm)

#checking accuracy
from sklearn.metrics import accuracy_score
knna = accuracy_score(Y_test,y_pred)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))


# # Using SVM model

# In[31]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[32]:


from sklearn import metrics                              # to check our accuracy,precision and recall by using our test set
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
cnf_matrix


# In[33]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))


# In[ ]:





# # Using MLPClassifier Model from neural_networks

# In[34]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100), random_state=1)
clf.fit(X_train,Y_train)


# In[35]:


clf.n_layers_


# In[36]:


np.shape(clf.coefs_[1])


# In[37]:


y_pred=clf.predict(X_test)
y_pred1=clf.predict(X_train)


# In[38]:


from sklearn import metrics                              # to check our accuracy,precision and recall by using our test set
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
cnf_matrix


# In[39]:


print("Accuracy:",metrics.accuracy_score(Y_train, y_pred1))


# In[40]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))


# In[ ]:





# # using RandomForestClassifier model

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train,Y_train)

y_predict_rf = rf_model.predict(X_test)


# In[43]:


from sklearn import metrics                              # to check our accuracy,precision and recall by using our test set
cnf_matrix = metrics.confusion_matrix(Y_test,y_predict_rf)
cnf_matrix


# In[44]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))


# In[45]:


# checking model accuracy 
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 
rf_r2_score = r2_score(Y_test, y_predict_rf)
print('R square Score = ', round(rf_r2_score, 3))

rf_mse = mean_squared_error(Y_test, y_predict_rf)
rf_rmse = np.sqrt(rf_mse)
print('Root Mean Squared Error = ', round(rf_rmse, 3))


# In[ ]:




