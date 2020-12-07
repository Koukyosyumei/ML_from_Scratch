#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from propencity_score import Propencity_Score


# In[2]:


biased_df = pd.read_csv("data/biased_data.csv")


# In[3]:


biased_df.head()


# In[4]:


biased_df["channel_multi"] = biased_df["channel"].apply(lambda x : 1 if x == "Multichannel" else 0)


# In[5]:


biased_df = biased_df.loc[:,["recency", "history", "treatment", "spend", "channel_multi"]]


# In[6]:


ps = Propencity_Score(biased_df, "spend", "treatment", bins=0.05)


# In[7]:


ps.get_coefficients()


# In[8]:


ps.plot_asam()


# In[ ]:




