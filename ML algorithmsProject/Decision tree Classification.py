#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#it is the process of grouping the samples:for eg- Pet images, cat images,etc.
# Algorithms; 1. Decision tree(Classification/Regression)
# svm- Knn-k means clustering-Random Forest- Association mining 


# # Information Gain : it is gain criteria used to estimate the information contained by each attribute.
# Entropy: It is impurity in the given dataset (Randomness)
# IG is the decrease in entropy and it is difference between entropy before split and average entropy after split.
# NOTE: the attribute with the highest IG is choose as the splitting attribute at the node.

# # Gini index: it is mainly applied on CART.
# 1. it is calculated for every subnodes.
# formula;- sum of the square of probabilities for success and failure.
# p= success
# q= failure  #P^2+q^2
# Gini= 0.25 =25% of the sample is hold the same result.
# * Gini score:- it is calculated using weighted gini score of each node of split.

# # Programming steps:
# 1. read the dataset using pandas
# 2. Features selections
# * separate X and Y
# 3. import decision tree classifier from sklearn.tree
# 4. train_test_split
# 5. fit the model
# 6. calculate metrics
# * Note: RMSE is mainly applicable for regressions.
# * for DT use accuracy measure
# 7. calculate accuracy_score
# 8. print the confusion matrix

# In[ ]:


import pandas as pd
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\diabetes.csv")
x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=df['Outcome']
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)   # function split
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)*100
print(acc)


# In[ ]:


res=model.predict([[6,148,72,35,0,33.6,0.627,50]])
print(res)


# In[ ]:


# Graphical Representation of Decision Tree
from sklearn.tree import export_graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


# In[ ]:


f=x.columns 
data=StringIO() # data read karne ke liye 
export_graphviz(model,out_file=data,filled=True,rounded=True,special_characters=True,feature_names=f,class_names=['0','1']) 
graph=pydotplus.graph_from_dot_data(data.getvalue())
graph.write_png('a.png') 
Image(graph.create_png())


# In[ ]:


p=model.predict([[0,150,100,0,0,25,0.5,25]])
print(p)


# In[ ]:


# Evaulation of Decision Tree
# 1. confusion matrix( popular)
# 2. precision and recall
# confusion matrix:
#TP : true positive  expected - belongs to class  ....Actual- belongs to class
# note- you were expecting A and result is alos A.

# TN: True Negative ... Expected: does not belongs to class....Actual: does not belongs to class
# FP: Flase positive...Expecting : Does not belongs to class....Actual: Belongs to class
# FN: Flase Negative....Expected: Belongs to class....Actual: not belongs to class


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


import matplotlib.pyplot as pt
import seaborn as sb
cmD=pd.DataFrame(cm,index=['1','0'],columns=['1','0'])
sb.heatmap(cmD,annot=True)
pt.ylabel('Actual Values')
pt.xlabel('Predicted Values')
pt.show()
df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})


# In[ ]:


print(df.to_string())


# # Precision and Recall:
# precision: what proportion of positive identification was actually correct.
# precision= TP/(TP+FP)
# NOTE: A model that produced no false positive has a precision of 1.
# Recall: what proportion of actual positive was identifed correctly.
# Recall= TP/(TP+FN)
# NOTE: A model that produces no false negatives has recall 1.

# In[ ]:


precision=cm[0,0]/[cm[0,0]+cm[0,1]]
print(precision)


# In[ ]:


recall=cm[0,0]/[cm[0,0]+cm[1,0]]
print(recall)


# In[ ]:




