#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\groceries - groceries.csv")
t=[]
for i in range(0,3000):
    t.append([str(df.values[i,j]) for j in range(0,18)])
    
print(t)


# In[ ]:


from apyori import apriori
trules=apriori(transactions=t,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
res=list(trules)
print(res)

