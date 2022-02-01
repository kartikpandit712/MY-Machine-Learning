#!/usr/bin/env python
# coding: utf-8

# In[1]:


#it is the concept of hyperplane
# hyperplane are decision boundaries those classified data points 
# support vectors: data points are closer to the hyperplane
# but the hyperplane which separates both the classes in proper manner is called as right hyperplane
# two type of svm: 1) Linear SVM   2) Non Linear SVM
#linear svm: in LSVM data is linearly separated.
# Non linear svm: In non linear SVM the classes are separated not by using line that can be  circle or polynomial.
# To identify the input it will compare the input with support vector to find the class.


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC   #support vector chain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\ads.csv")
#gmap={"Male":0,"Female":1}
#df['Gender']=df['Gender'].map(gmap)
x=df[['Age','EstimatedSalary']]
y=df['Purchased']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
scaleX=StandardScaler()
x_train=scaleX.fit_transform(x_train)
x_test=scaleX.fit_transform(x_test)
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[3]:



x_set, y_set=x_train, y_train
#print(x_set[:1].min()-1)
X1,X2=np.meshgrid(np.arange(start=x_set[:, 0].min()-1,stop=x_set[:, 0].max()+1,step=0.01),np.arange(start=x_set[:, 1].min()-1,stop=x_set[:, 1].max()
+1,step=0.01))
pt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75, cmap = ListedColormap('red','green'))
pt.xlim(X1.min(),X1.max())
pt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    pt.scatter(x_set[y_set==j,1],x_set[y_set==j,i])

pt.show()


# In[ ]:




