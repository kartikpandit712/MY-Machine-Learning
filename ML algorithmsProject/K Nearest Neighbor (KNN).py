#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Two type of algorithm based on model fitting (training)
# 1. Eager(early)- they fit and build a model before prediction.
# 2.Lazy- they fit the model at the time of prediction.
 # And the KNN is Lazy Algorithm that why it is most prefered.
# It can handle realtime database so there is no issue with changes in databse.
 # we find the distance between points using distance measures such as euclidean distance.
    
    #ED- 1. calculate the distance
    # 2. find the closet neighbours 
    # 3. vote for labels.


# # Curse of dimensionality:  major disadvantage of KNN
# since the data is dynamically added for featurs are dynamically changes it may possible that algorithm increases its complexity exponentially this is called as curse of dimensionality.

# # Solution : Cosine Similarity
# if new data is comming and it already exists in the file then do not add that data. Bcoz we already have that one.

# # How to decide K: k should be odd if the number of classes are even.  K=hyperparameter
# Note: most of the libraries uses Eucledian distance for deciding measures.

# # steps: 
# 1.sklearn.neighbors-KneighborsClassifier
# 2.fit transform if neccessary + spliting
# 3.fit the model
# 4.predict
# Note: Increase K to increase accuracy (depends on dataset)

# In[ ]:


import numpy as np 
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\ads.csv")
x=df[['Age','EstimatedSalary']]
y=df['Purchased']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
model=KNeighborsClassifier(n_neighbors=3)
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
x1,x2=np.meshgrid(np.arange(start=x_set[:, 0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:, 1].min()-1,stop=x_set[:, 1].max()+1,step=0.01))
pt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','blue')))
pt.xlim(x1.min(),x1.max())
pt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    pt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],cmap=ListedColormap(('red','blue'))(i),label=j)
    
pt.show()
    


# In[ ]:





# In[ ]:




