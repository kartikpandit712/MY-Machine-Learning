#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The RF is used to overcome the Decision Tree overfitting disavantage.
# RF is a classifier which contains a numbers of DT.
# greater no. of trees higher the accuracy.


# In[ ]:


# implementation steps: 1.data preprocessing step
# 2. fitting the random forest algorithm to train set
# 3. predict the test result
# 4. test the accuracy
# 5.confusion matrix


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\ads.csv")
x=df[['Age','EstimatedSalary']]
y=df['Purchased']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
model=RandomForestClassifier(n_estimators=8,criterion='entropy')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
res=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(res.to_string())


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as pt
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:, 0].min()-1,stop=x_set[:, 0].max()+1,step=0.01),np.arange(start=x_set[:, 1].min()-1,stop=x_set[:, 1].max()+1,step=0.01))
pt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.7,cmap=ListedColormap(('red','green')))
pt.xlim(x1.min(),x1.max())
pt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    pt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap('red','green'))
    pt.show()


# In[ ]:




