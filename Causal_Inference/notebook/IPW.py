#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[2]:


biased_df = pd.read_csv("data/biased_data.csv")
biased_df["channel_multi"] = biased_df["channel"].apply(lambda x : 1 if x == "Multichannel" else 0)
biased_df = biased_df.loc[:,["recency", "history", "treatment", "spend", "channel_multi"]]


# In[3]:


target = biased_df.spend.values


# In[4]:


biased_covariates = biased_df.drop(
            ["treatment", "spend"], axis=1)

biased_treatment = biased_df["treatment"]


# In[5]:


glm = LogisticRegression()
glm.fit(biased_covariates, biased_treatment)
propencity_score = glm.predict_proba(biased_covariates)[:, 1]


# In[6]:


#propencity_score


# In[7]:


#biased_df["score"] = propencity_score


# In[8]:


z = biased_df.treatment.values
y = biased_df.spend.values
ps = propencity_score

#ipwe1 = sum((z*y)/ps)/sum(z/ps)
#ipwe0 = sum(((1-z)*y)/(1-ps))/sum((1-z)/(1-ps))


# In[9]:


explain_var = biased_df.drop("treatment", axis=1).values


# In[10]:


n = explain_var.shape[0]


# In[11]:


c1_ipwe1_adj = sum(np.multiply((z/ps).reshape(n, 1), explain_var)) / sum(z/ps)                   
c1_ipwe0_adj = sum(np.multiply(((1-z)/(1-ps)).reshape(n, 1), explain_var)) / sum((1-z)/(1-ps))
c1_ipwe1_adj - c1_ipwe0_adj


# In[12]:


c1_ipwe1 = sum(np.multiply((z).reshape(n, 1), explain_var)) / sum(z)      
c1_ipwe0 = sum(np.multiply((1-z).reshape(n, 1), explain_var)) / sum(1-z)
c1_ipwe1 - c1_ipwe0


# In[ ]:





# In[ ]:




