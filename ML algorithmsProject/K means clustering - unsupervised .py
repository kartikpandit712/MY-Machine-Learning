#!/usr/bin/env python
# coding: utf-8

# *it is unsupervised and centroid based algorithm.
# *data is seperated based on the centers and nature.
# *it is iterative process where centroid may get change until the data is grouped properly.
# *The main aim of alogrithm is to minimize the sum of distance between the data points and their clusters.
# *K is hyperparameter.
# 

# # working:
# 1.select the number of K to decide the number of clusters
# 2.select the random K points or centroids 
# 3.assign each data points to their closest centroid 
# 4.calculate the variance and place new centroid
# 5.repeat the third steps
# 6.if any assigment occurs then goto 4th step else stop
# 7.model ready

# # Elbow method: to decide the K
# *within the cluster sum of square(WCSS)
# 1.it executes the k means clustering on given data set for different k values.
# 2.for each K calculate WCSS
# 3.plot the curve betn wcss
# 4.identify the elbow(bend) that is considered as K value.
# Note: if there are more than one elbow then use elbow with max"K".

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\Mall_Customers.csv")
#genderMap={"Male":1,"Female":0}
#df['Genre']=df['Genre'].map(genderMap)
x=df[['Annual Income (k$)','Spending Score (1-100)']]
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
pt.plot(range(1,11),wcss)
pt.xlabel("K")
pt.ylabel("WCSS")
pt.show()


# In[5]:


model=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_pred=model.fit_predict(x)
print(y_pred)


# In[7]:



#pt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100,c='red',label="C1")
#pt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100,c='green',label="C2")
#pt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100,c='blue',label="C3")
#pt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100,c='yellow',label="C4")
#pt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100,c='cyan',label="C5")
pt.scatter(model.cluster_centers_[:, 0],model.cluster_centers_[:, 1],s=200,c='pink',label="center")
pt.show()


# In[ ]:




