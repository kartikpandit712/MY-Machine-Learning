#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix as CM
import seaborn as sns


# In[2]:


import pandas as pd
data =pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\FML\short-assignment-2-kartikpandit712-main\crab.txt",delimiter="\t")
print(data)


# In[3]:


features=data[['FrontalLip','RearWidth','Length','Width','Depth','Male','Female']]
sp=data['Species']
print(sp)


# In[4]:


features = np.array(data.iloc[:,1:]) # training data location 
sp = np.array (data.iloc[:,0])  # target data location
print(features)


# In[5]:


#x_train = data.iloc[:140,1:]#.to_numpy()
#t_train = data.iloc[:140,0]#.to_numpy()

#x_test = data.iloc[140:,1:]#.to_numpy()
#t_test = data.iloc[140:,0]#.to_numpy()

#x_train.shape, x_test.shape, t_train.shape, t_test.shape


# In[6]:


x_train,x_test,t_train,t_test=train_test_split(features,sp,test_size=0.3,random_state=3)
x_train# X_train for train data and t_train for target data


# In[7]:


train =  data.iloc[:140,:]#
train


# In[ ]:





# In[8]:


train[train['Species'] == 0]


# In[9]:


train[train['Species'] == 1]


# In[ ]:





# In[10]:


#ct = ColumnTransformer([('first 5 attributes',StandardScaler(),[i for i in range(5)])],remainder = 'passthrough')
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import StandardScaler

# This pipeline will apply Standardization to all numerical attributes 
# The attributes that are one-hot/interger-encoded (such as gender) will remain as is

#scaling_pipeline = ColumnTransformer([('num_attribs', StandardScaler(), list(range(5)))],
                                    #remainder='passthrough')
#scaling_pipeline


# In[11]:


#scaling_pipeline.fit(x_train)

# The fit() method learns the necessary parameters of the pipeline.
# Since this pipeline includes preprocessing scalars, it will learn the mean and std for each attriute
# If it were a classifier/regression, it will train all the parameters associated with that classifier/regression algorithm


# In[12]:


CT = ColumnTransformer([('first 5 attributes',StandardScaler(),[i for i in range(5)])],remainder = 'passthrough')


# In[13]:


x_train_transform = CT.fit_transform(x_train)
x_test_transform = CT.transform(x_test)


# In[14]:


### Estimate Parameters

# means for class_1 and class_2
mean1 = np.mean(x_train_transform[np.where(t_train == 0)[0]],axis = 0)
mean2 = np.mean(x_train_transform[np.where(t_train == 1)[0]],axis = 0)

cov1 = np.cov(x_train_transform[np.where(t_train == 0)[0]].T)
cov2 = np.cov(x_train_transform[np.where(t_train == 1)[0]].T)

cov1 += np.eye(cov1.shape[0])*(1e-5)
cov2 += np.eye(cov2.shape[0])*(1e-5)


# In[15]:


#Estimate prior probabilities
P1 = np.count_nonzero(t_train == 0)/t_train.shape[0]
P2 = np.count_nonzero(t_train == 1)/t_train.shape[0]


# In[22]:


# for training dataset
x= [1, 13, 14, 15, 11, 23, 35]
t1_x1 = multivariate_normal.pdf(x = x_train_transform, mean = mean1,cov = cov1)
t2_x1 = multivariate_normal.pdf(x = x_train_transform, mean = mean2,cov = cov2)

print('Data likelihoods:')
print('P(x|C1) = ',t1_x1)
print('P(x|C2) = ', t2_x1,'\n')

t1_pos = t1_x1*P1/(t1_x1*P1 + t2_x1*P2)
t2_pos = t2_x1*P2/(t1_x1*P1 + t2_x1*P2)

print('Posterior probabilities:')
print('P(C1|x) = ', t1_pos)
print('P(C2|x) = ', t2_pos,'\n')

if t1_pos > t2_pos:
    print('x = ',x,' belongs to class 1')
else:
    print('x = ',x,' belongs to class 2')

t_Prediction = (t2_pos>t1_pos)*1

CM_train = CM(t_train,t_Prediction)
print(CM_train)


# In[ ]:


res = sns.heatmap(CM_train, annot=True, cmap='Reds')

res.xaxis.set_ticklabels(['1','0'])
res.yaxis.set_ticklabels(['0','1'])
plt.show()


# In[20]:


# for test dataset

t1_x1 = multivariate_normal.pdf(x = x_test_transform, mean = mean1,cov = cov1)
t2_x1 = multivariate_normal.pdf(x = x_test_transform, mean = mean2,cov = cov2)

print('Data likelihoods:')
print('P(x|C1) = ', t1_x1)
print('P(x|C2) = ', t2_x1,'\n')

t1_pos = t1_x1*P1/(t1_x1*P1 + t2_x1*P2)
t2_pos = t2_x1*P2/(t1_x1*P1 + t2_x1*P2)

print('Posterior probabilities:')
print('P(C1|x) = ', t1_pos)
print('P(C2|x) = ', t2_pos,'\n')

if t1_pos > t2_pos:
    print('x = ',x,' belongs to class 1')
else:
    print('x = ',x,' belongs to class 2')
    
t_prediction = (t2_pos>t1_pos)*1

CM_test = CM(t_test,t_prediction)
print(CM_test)


# In[ ]:


res = sns.heatmap(CM_train, annot=True, cmap='Reds')

res.xaxis.set_ticklabels(['1','0'])
res.yaxis.set_ticklabels(['0','1'])
plt.show()


# In[ ]:





# Answer 2
# 

# While Solving the problem it enconter the singular matrix error. As per my understanding out of seven features male and female are binary and rest and numerical. So, initailly I using all the features which say our matrix was singular beacuse of male andfemale. So, making it singular I select one feature out of two.

# In[ ]:




