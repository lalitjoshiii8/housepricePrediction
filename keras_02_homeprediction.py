#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[63]:


df = pd.read_csv('kc_house_data.csv')


# In[64]:


df.isnull().sum()


# In[65]:


df.describe().transpose()


# In[66]:


plt.figure(figsize=(10,6))
sns.distplot(df['price'])


# In[67]:


sns.countplot(df['bedrooms'])


# In[68]:


df.corr()['price'].sort_values()


# In[69]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='price' , y='sqft_living', data = df)


# In[70]:


plt.figure(figsize=(10,5))
sns.boxplot(x='bedrooms' , y='price',data=df)


# In[71]:


df.columns


# In[72]:


sns.scatterplot(x='price' , y='long' , data=df)


# In[73]:


sns.scatterplot(x='price' , y='lat' , data=df)


# In[74]:


sns.scatterplot(x='long' , y='lat' , data=df)


# In[75]:


df.sort_values('price' , ascending=False).head(20)


# In[76]:


non_top_1_perc = df.sort_values('price' , ascending=False).iloc[216:]


# In[77]:


sns.scatterplot(x='long' , y='lat' , data=non_top_1_perc , hue = 'price' , edgecolor=None , alpha=0.2)


# In[78]:


df = df.drop('id' , axis=1)


# In[79]:


df['date'] = pd.to_datetime(df['date'])
df['date']


# In[80]:


def year_extraction(date):
    return date.year


# In[81]:


df['year']=df['date'].apply(lambda date: date.year)


# In[82]:


df['month']=df['date'].apply(lambda date: date.month)


# In[83]:


df.head()


# In[84]:


sns.boxplot(x='month' , y = 'price' , data=df)


# In[85]:


df.groupby('year').mean()['price'].plot()


# In[86]:


df = df.drop('date' , axis=1)


# In[87]:


df.columns


# In[88]:


df.head()


# In[89]:


df['zipcode'].value_counts()


# In[90]:


df=df.drop('zipcode' , axis=1)


# In[91]:


df['yr_renovated'].value_counts()


# In[92]:


df['sqft_basement'].value_counts()


# In[93]:


X=df.drop('price',axis=1)
y=df['price'].values


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.30 , random_state=102)


# In[96]:


from sklearn.preprocessing import MinMaxScaler


# In[97]:


scaler = MinMaxScaler()


# In[98]:


X_train = scaler.fit_transform(X_train)


# In[99]:


X_test = scaler.transform(X_test)


# In[101]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[102]:


X_train.shape


# In[130]:


model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')


# In[131]:


model.fit(x = X_train , y = y_train , validation_data=(X_test,y_test),
         batch_size=128,epochs=1500)


# In[132]:


losses = pd.DataFrame(model.history.history)


# In[133]:


losses.plot()


# In[134]:


from sklearn.metrics import mean_squared_error , mean_absolute_error , explained_variance_score


# In[135]:


predictions = model.predict(X_test)


# In[146]:


np.sqrt(mean_squared_error(y_test , predictions))


# In[137]:


df['price'].describe()


# In[138]:


explained_variance_score(y_test , predictions)


# In[139]:


plt.scatter(y_test,predictions)
plt.plot(y_test , y_test,'r')


# In[140]:


single_house = df.drop('price',axis=1).iloc[0]


# In[141]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# In[142]:


model.predict(single_house)


# In[143]:


df.head(1)


# In[ ]:




