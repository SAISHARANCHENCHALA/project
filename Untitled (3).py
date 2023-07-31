#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.read_csv('greendestination.csv')


# In[10]:


df


# In[16]:


df.info()


# In[15]:


df=pd.get_dummies(df,drop_first=True)
df


# In[17]:


X=df.drop('Attrition_Yes',axis=1)
y=df['Attrition_Yes']


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[20]:


scaler = StandardScaler()


# In[21]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


log_model = LogisticRegression()


# In[24]:


log_model.fit(scaled_X_train,y_train)


# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix


# In[26]:


y_pred = log_model.predict(scaled_X_test)


# In[27]:


confusion_matrix(y_test,y_pred)


# In[28]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[29]:


accuracy_score(y_test,y_pred)


# In[30]:


log_model.coef_


# In[31]:


sns.pairplot(df,hue='Attrition_Yes')


# [Tableau](https://public.tableau.com/app/profile/sai.sharan5894/viz/project1_16908223350690/Sheet1?publish=yes)

# https://public.tableau.com/app/profile/sai.sharan5894/viz/project1-1_16908236237850/Sheet1?publish=yes

# https://public.tableau.com/app/profile/sai.sharan5894/viz/project1-2_16908244750280/Sheet1?publish=yes

# https://public.tableau.com/app/profile/sai.sharan5894/viz/project1-3_16908246471110/Sheet1?publish=yes

# In[36]:


df2=pd.read_csv('Amazon Sales data.csv')


# In[37]:


df2


# In[38]:


df2.info()


# In[ ]:




