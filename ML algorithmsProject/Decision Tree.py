#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# there is a root node, every node holds some conditions, if condition is true then goto left side otherwise right side.The outcome is at the leaf node.
# Note: for DT alll data should be numerical if not then convert into numerical, and it handle by pandas.


# In[ ]:


# spliting the data: Gini index is the formula based factor to split the data or sample. 1-(x/n)^2 - (y/n)^2
# x= -ve result and Y=+ve result, no= no. of sample.
# Gini is o then result are same for all sample. gini=0.25 of the sample will hold same result.


# In[ ]:


# convert to numeric
import pandas as pd 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\drug200.csv")
print(df)


# In[ ]:


# Mapping
sm={'F':0,'M':1}
df['Sex']=df['Sex'].map(sm)
print(df.head())


# In[ ]:


bpm={'HIGH':1,'NORMAL':0,'LOW':2}
df['BP']=df['BP'].map(bpm)
print(df.head())


# In[ ]:


cm={'HIGH':1,'NORMAL':0}
df['Cholesterol']=df['Cholesterol'].map(cm)
print(df.head())


# In[ ]:


dm={'drugA':1,'drugB':2,'drugC':3,'drugX':4,'drugY':5}
df['Drug']=df['Drug'].map(dm)
print(df.head())


# In[ ]:


Features=['Age','Sex','BP','Cholesterol','Na_to_K']
x=df[Features]
y=df['Drug']
print(x)
print(y)


# In[ ]:


dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
result=dtree.predict([[23,0,1,1,25.355]])
print(result)


# In[ ]:


result=dtree.predict([[47,1,2,1,10.114]])
print(result)


# In[ ]:




