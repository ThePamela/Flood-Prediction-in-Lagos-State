#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


# In[3]:


df = pd.read_csv("C:\\Users\\user\\Documents\\doc\\Rain Data in Lagos.csv")


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce') 
df


# In[8]:


plt.figure(figsize=(10,6))
x = df["Date"]
y = df["Precipitation"]
plt.xlabel("Date")
plt.ylabel("Precipitation (m)")
plt.title('Daily Flood in Lagos ')
plt.plot(x,y)
plt.show()


# In[9]:


df['Date'] = pd.to_datetime(df['Date'])


# In[12]:


df.set_index('Date', inplace=True)


# In[15]:


monthly_rain_data = df['Precipitation'].resample('ME').sum()


# In[16]:


plt.figure(figsize=(10, 6))
monthly_rain_data.plot()
plt.title('Monthly Precipitation in Lagos State')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.show()


# In[17]:


from statsmodels.tsa.arima.model import ARIMA


# In[18]:


train_data = monthly_rain_data[:int(0.8 * len(monthly_rain_data))]
test_data = monthly_rain_data[int(0.8 * len(monthly_rain_data)):]


# In[19]:


model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()


# In[22]:


Forecast = model_fit.forecast(steps=12)


# In[23]:


plt.plot(monthly_rain_data, label='History Record')
plt.plot(Forecast, label='Forecast', color='purple')
plt.title('Monthly Precipitation Forecast')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.show()


# In[31]:


flood_threshold = 300


# In[32]:


flood_months = Forecast[Forecast > flood_threshold]


# In[33]:


print("Predicted flood months:")
print(flood_months)


# In[ ]:




