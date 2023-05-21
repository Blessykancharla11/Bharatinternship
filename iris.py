#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[7]:


iris = pd.read_csv('/Users/blessykancharla/Downloads/Iris.csv')


# In[8]:


iris.head()


# In[9]:


iris.describe()


# In[13]:


iris.shape


# In[14]:


iris.info()


# In[16]:


iris


# In[22]:


iris.dtypes


# In[23]:


iris.plot()


# In[24]:


sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
plt.show()


# In[25]:


sns.pairplot(iris,hue='Species')


# In[27]:


iris['SepalLengthCm'].hist()
plt.title('Histogram')


# In[28]:


iris['PetalLengthCm'].hist()
plt.title('Histogram')


# In[29]:


iris['PetalWidthCm'].hist()
plt.title('Histogram')


# In[30]:


iris['Species'].hist()
plt.title('Histogram')


# In[36]:


colors = ['red','blue','green']
species = ['Iris-setosa','Iris-virginica','Iris-versicolor']
for i in range(3):
    x= iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'],x['SepalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("SepalLengthCm")    
plt.ylabel("SepalWidthCm") 


# In[35]:


colors = ['red','blue','green']
species = ['Iris-setosa','Iris-virginica','Iris-versicolor']
for i in range(3):
    x= iris[iris['Species'] == species[i]]
    plt.scatter(x['PetalWidthCm'],x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("PetalLengthCm")    
plt.ylabel("PetalWidthCm") 


# In[38]:


colors = ['red','blue','green']
species = ['Iris-setosa','Iris-virginica','Iris-versicolor']
for i in range(3):
    x= iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("SepalLengthCm")    
plt.ylabel("PetalWidthCm") 


# In[39]:


colors = ['red','blue','green']
species = ['Iris-setosa','Iris-virginica','Iris-versicolor']
for i in range(3):
    x= iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'],x['SepalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("SepalWidthCm")    
plt.ylabel("PetalLengthCm") 


# In[61]:


X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[66]:


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test)*100)


# In[ ]:




