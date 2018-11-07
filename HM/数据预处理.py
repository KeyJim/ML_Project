
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import DataFrame


# In[30]:


dframe = pd.read_excel("test.xlsx")


# In[31]:


dframe


# In[32]:


test = DataFrame(dframe)


# In[36]:


test=test.T


# In[37]:


test


# In[38]:


X = test.iloc[ : , :-1].values 


# In[39]:


X


# In[41]:


Y = test.iloc[ : , -1].values 


# In[42]:


Y


# In[10]:


plt.plot(test4)

