#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[8]:


breast_cancer = load_breast_cancer()


# In[9]:


X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
y = pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)


# In[12]:


encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))


# In[13]:


train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state = 1)


# In[15]:


classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
classifier.fit(train_X,train_y)


# In[16]:


prediction = classifier.predict(test_X)


# In[17]:


confusion_matrix(test_y,prediction)


# In[19]:


accuracy  = accuracy_score(test_y,prediction)
print('AdaBoost Accuracy:',accuracy)


# In[ ]:




