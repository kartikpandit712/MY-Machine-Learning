#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This algorithms predicts the future value based on dataset.
# Different types of regression.
# 1. Linear Regression
# 2. Multilinear Regression
# 3. Polynomial Regression


# # Linear Regression : y=mx+c
# one dependent and one independent variable

# # Multilinear Regression: y= ax1+bx2+c
# one dependent and many independent

# # Polynomial Regression: y= mx^n+c
# it is more than one degree of linear regression

# In[ ]:


import numpy as np
a=[1,2,3,4,5]
per=np.percentile(a,50)
print(per)


# In[ ]:


# Linear Regression   scipy under stats
# outcome of algorithms
# r= coef correlation
# p= probabilistic error value
# std_err= standard error
# r = -1 to 1
# 0= no correlation
# 1= strong positive relation 
# -1 = negative relation
# threshold of r:(-0.2 or 0.2)
# r<-0.2 relation is possible
# r> 0.2 relation is possible


# In[ ]:


from scipy import stats
x=[1,2,3,4,5]
y=[2,4,6,8,10]
m,c,r,p,std_err=stats.linregress(x,y)

def predict(x):
    return m*x+c
print("y value")
yVal=predict(6)
print(yVal)


# In[ ]:


from scipy import stats
import matplotlib.pyplot as pt
x=[1,2,3,4,5]
y=[4,8,12,16,20]
m,c,r,p,std_err=stats.linregress(x,y)
model=list(map(predict,x))
pt.scatter(x,y)
pt.plot(x,model)
pt.show()


# In[ ]:


# Note : if 'r' is not in our thershold then follow polynomial regression


# In[ ]:


# polynomial regression - Numpy
# y= mx^2+c
import numpy as np
import matplotlib.pyplot as pt
x=[2001,2002,2003,2004]
y=[5000,2000,10000,3000]
model=np.poly1d(np.polyfit(x,y,3))   # 3-degree
yVal=model(2005)
print(yVal)
line=np.linspace(1,2004,3000)
pt.plot(line,model(line))
pt.show()


# In[ ]:


# Verifying polynomial regression
   # r-squared score: SKlearn metrics -- it values lies between 0-1
# if (r^2 > 0.2) some sort of regression exists.


# In[ ]:


import numpy as np 
from sklearn.metrics import r2_score
x=[1,2,3,4,5,6,7,8,9,10]
y=[10,13,7,18,6,15,18,12,0,20]    #polyfit
model=np.poly1d(np.polyfit(x,y,3))
score=r2_score(y,model(x))
print(score)


# In[ ]:


# Multilinear Regression : for reading data set (use pandas)  ( consider y as output)
# dataset main last wala part y as [output] consider hota hai.


# In[ ]:


import pandas as pd
from sklearn import linear_model
df=pd.read_csv(r"C:\Users\karti\Dropbox\PC\Downloads\cars.csv")  #dataframe
x=df[['Volume','Weight']]   #independent
y=df['CO2']
reg=linear_model.LinearRegression()
reg.fit(x,y)
predict=reg.predict([[2500,3000]])
print(predict)


# In[ ]:


#multilinear Regression:
# train_test_split:
# for a given dataset split it into 2 parts
#training: testing= 70% : 30% 
# note: the training set should be greater than 50%.
# Evaulating model: (check the errors)
#1.mean absolute error, 2. mean squared error, 3.root mean squared error: decision maker
# RMSE= it should be less than the 10% of the mean value of the output op mean=70, 10% op=70, RMSE=6 acceptable, RMSE= 10 not acceptable


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
df=pd.read_csv(r"‪C:\Users\karti\Dropbox\PC\Downloads\petrol.csv")
x=df[['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence']]
y=df['Petrol_Consumption']
print(df.describe())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(df)
mae=metrics.mean_absolute_error(y_test,y_pred)
print(mae)
mse=metrics.mean_square_error(y_test,y_pred)
print(mse)
rmse=np.sqrt(mse)
print(rmse)


# In[ ]:




