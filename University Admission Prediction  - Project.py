#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'C:\Users\Piyush\Desktop\Admission_Predict_Ver1.1.csv')


# In[3]:


df.head()


# In[4]:


df.drop(['Serial No.'],axis = 1,inplace = True)
df.head()


# # Visualizing Data to analyze Predictions

# In[5]:


df.hist(bins = 20,figsize = (20,20),color = 'red')


# In[6]:


import seaborn as sns


# In[7]:


sns.pairplot( df)


# # Testing and Traing data

# In[20]:


x = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']]
y = df['Chance of Admit ']
xd = np.array(x)
yd = np.array(y)

yd = yd.reshape(-1,1)


# In[21]:


#Scaling the data before training - because different features have different max. and min and dont want model to be biased
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler_x = StandardScaler()
xd = scaler_x.fit_transform(xd)
scaler_y = StandardScaler()
yd = scaler_y.fit_transform(yd)


# In[10]:


#  splitting the test and train dataset
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test  = tts(x,y,test_size = 0.15)


# # Multiple Linear Regression Model

# In[11]:


# Multiple linear Regression
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x_train,y_train)
y_predict = lin.predict(x_test)


# In[12]:


# Checking the accuracy
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
lin_accuracy = lin.score(x_test,y_test)
lin_accuracy


# In[13]:


#visualizing through regression plot - only valid to see relationship between one ind. variable with our y
width =  10
height = 10
plt.figure(figsize = (width,height))
sns.regplot(x = 'GRE Score',y = 'Chance of Admit ',data =df)


# In[14]:


# hence for multiple linear regression it is better to visualize through distribution plots 
plt.figure(figsize = (width,height))
sns.distplot(y_test,hist = False,color = 'r')
sns.distplot(lin.predict(x_test),hist = False,color = 'g')


# In[16]:


#getting back the original values from scaling
#y_orig_test = scaler_y.inverse_transform(y_test)
#y_orig_predict = scaler_y.inverse_transform(lin.predict(x_test))


# In[18]:


#visualizig for original values
#plt.figure(figsize = (width,height))
#sns.distplot(y_orig_test,hist = False,color = 'r')
#sns.distplot(y_orig_predict,hist = False,color = 'g')


# In[19]:


#visualizing using actual and predicted values
plt.plot(y_test,y_predict,'*',color = 'r')


# In[ ]:


#plot for original valiues
#plt.plot(y_orig_test,y_orig_predict,'*',color = 'g')


# # Calculating RMSE and R^2 

# In[22]:


from sklearn.metrics import mean_squared_error,r2_score 


# In[24]:


lin_mse = np.sqrt(mean_squared_error(y_test,y_predict))
lin_r2 = r2_score(y_test,y_predict)
print(lin_mse)
print(lin_r2)


# In[25]:


lin_accuracy = lin.score(x_test,y_test)
print('Accuracy: ',lin.score(x_test,y_test))


# # Polynomial Regression Model

# In[26]:


from sklearn.preprocessing import PolynomialFeatures 


# In[29]:


poly = PolynomialFeatures(2)  # say degree 2
x_poly = poly.fit_transform(x_train)
lin_poly = LinearRegression()
lin_poly.fit(x_poly,y_train)
y_poly_predict = lin_poly.predict(poly.fit_transform(x_test))


# In[30]:


plt.figure(figsize = (width,height))
sns.regplot(x = 'GRE Score',y = 'Chance of Admit ',data =df,order = 2)


# In[32]:


plt.plot(y_test,y_poly_predict,'^',color = 'r')


# In[33]:


#y_poly_orig_predict = scaler_y.inverse_transform(y_poly_predict)
#plt.plot(y_orig_test,y_poly_orig_predict,'^',color = 'g')


# In[35]:


poly_mse = np.sqrt(mean_squared_error(y_test,y_poly_predict))
poly_r2 = r2_score(y_test,y_poly_predict)
print(poly_mse)
print(poly_r2)


# In[ ]:





# # Decison Tree

# In[36]:


# #Training and Evaluating Decison Trees
from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor()
dec_tree.fit(x_train,y_train)
plt.plot(y_test,dec_tree.predict(x_test),'o',color = 'g')


# In[37]:


dec_tree_accuracy = dec_tree.score(x_test,y_test) 
dec_tree_accuracy


# In[ ]:


# for decison tree model
#y_tree_orig_predict = scaler_y.inverse_transform(dec_tree.predict(x_test))
#plt.plot(y_orig_test,y_tree_orig_predict,'o',color = 'g')


# In[51]:


dt_error = np.sqrt(mean_squared_error(y_test,dec_tree.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,dec_tree.predict(x_test))))
print(r2_score(y_test,dec_tree.predict(x_test)))


# # Random Forest

# In[41]:


from sklearn.ensemble import RandomForestRegressor
rand_forest = RandomForestRegressor(n_estimators = 100,max_depth = 10)
rand_forest.fit(x_train,y_train)


# In[42]:


rand_for_accuracy = rand_forest.score(x_test,y_test)
rand_for_accuracy


# In[53]:


f_error = np.sqrt(mean_squared_error(y_test,rand_forest.predict(x_test)))
f_error


# # Conclusion

# ### R^2 is highest for Multiple linear regression and  worst being of Decsion Trees
#    

# In[54]:


print('R^2 Score: ')
print('Multiple Linear Regression Model: ',lin_accuracy)
print('Plolynomial Regression Model: ',poly_r2)
print('Decision Tree Model: ',dec_tree_accuracy)
print('Randon Foest Model: ',rand_for_accuracy)


# In[56]:


print('Root Mean Squared Error Score: ')
print('Multiple Linear Regression Model: ',lin_mse)
print('Plolynomial Regression Model: ',poly_mse)
print('Decision Tree Model: ',dt_error)
print('Randon Foest Model: ',f_error)


# ## Accuracy: Multiple Linear > Polynomial > Random Forest > Decision Tree
