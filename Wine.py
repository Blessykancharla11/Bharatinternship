#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# In[12]:


import warnings
warnings.filterwarnings('ignore')


# In[99]:


df = pd.read_csv('/Users/blessykancharla/Downloads/winequality-red.csv')
print(df.head())


# In[101]:


df.info()


# In[102]:


df.describe().T


# In[103]:


df.nunique()


# In[129]:


sns.set(style="whitegrid")
print(df['quality'].value_counts())
fig = plt.figure(figsize = (7,5))
sns.countplot( x = 'quality', data=df, palette='plasma')


# In[131]:


sns.set(style="whitegrid")
fig, ax1 = plt.subplots(3,4, figsize=(24,30))
k = 0
columns = list(df.columns)
for i in range(3):
    for j in range(4):
            sns.boxplot(x='quality', y=columns[k], data=df, ax=ax1[i][j], palette='deep')
            k += 1
plt.show()


# In[132]:


plt.figure(figsize = (15,15))
sns.heatmap(df.corr(),annot=True, cmap= 'magma')


# In[107]:


df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[108]:


df.corr()['quality'].sort_values(ascending=False)


# In[109]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[110]:


plt.figure(figsize=(9, 9))
sns.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[111]:


df = df.drop('total sulfur dioxide', axis=1)


# In[94]:


df.columns


# In[112]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)


# In[113]:


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape


# In[114]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[115]:


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()


# In[122]:


y_pred = models[1].predict(xtest)  # Assuming models[1] is your trained model
cm = confusion_matrix(ytest, y_pred)
print(cm)


# In[133]:


print(metrics.classification_report(ytest,
									models[1].predict(xtest)))


# Conclusion 
#     With all the ananlysis the parameters which are important in finding the good wine are alcohol, volatile acidity, sulphates, density, total_sulfur_dioxide, and they will improve the wine quality.
# 
