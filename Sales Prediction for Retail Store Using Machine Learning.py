#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[55]:


train = pd.read_csv(r"C:\Users\revat\Downloads\archive (16)\train.csv")
store = pd.read_csv(r"C:\Users\revat\Downloads\archive (16)\store.csv")


# In[56]:


df


# In[57]:


df.head(5)


# In[58]:


df.tail(5)


# In[59]:


df.describe()


# In[60]:


df.columns


# In[61]:


df.columns.tolist()


# In[62]:


df.info()


# In[63]:


df.sample(5)


# In[64]:


df.shape


# In[65]:


df.dtypes


# In[66]:


df.isnull().sum()


# In[67]:


df.duplicated().sum()


# In[68]:


df.fillna(0,inplace=True)


# In[70]:


df = pd.merge(train, store, on="Store", how="left")


# In[72]:


df.columns


# In[73]:


train.columns


# In[74]:


df.columns


# In[79]:


print(df["StateHoliday"].unique())
print(df["StateHoliday"].dtype)


# In[99]:


le = LabelEncoder()

df["StoreType"] = le.fit_transform(df["StoreType"])
df["Assortment"] = le.fit_transform(df["Assortment"])
df["StateHoliday"] = le.fit_transform(df["StateHoliday"])
df["PromoInterval"] = le.fit_transform(df["PromoInterval"].astype(str))


# In[103]:


df.columns.tolist()


# In[112]:


print(train.columns.tolist())
print(store.columns.tolist())


# In[113]:


df = pd.merge(train, store, on="Store", how="left")


# In[114]:


print(df.columns.tolist())


# In[116]:


df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# In[117]:


df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)

df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)

df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)

df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

df["PromoInterval"] = df["PromoInterval"].fillna("None")


# In[118]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["StoreType"] = le.fit_transform(df["StoreType"].astype(str))
df["Assortment"] = le.fit_transform(df["Assortment"].astype(str))
df["StateHoliday"] = le.fit_transform(df["StateHoliday"].astype(str))
df["PromoInterval"] = le.fit_transform(df["PromoInterval"].astype(str))


# In[119]:


X = df[
[
    "DayOfWeek",
    "Customers",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "CompetitionDistance",
    "Promo2",
    "PromoInterval",
    "Month",
    "Year"
]]

y = df["Sales"]


# In[121]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[123]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[125]:


y_pred = lr.predict(X_test)


# In[126]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))


# In[127]:


comparison = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

comparison.head(10)


# In[134]:


from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)


# In[135]:


xgb_pred = xgb.predict(X_test)


# In[136]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# In[137]:


df["Promo"].value_counts()


# In[140]:


plt.figure(figsize=(6,5))
plt.scatter(df["Customers"], df["Sales"], color="pink")

plt.title("Customers vs Sales")
plt.xlabel("Customers")
plt.ylabel("Sales")

plt.show()


# In[141]:


X["Promo"]


# In[143]:


df["StateHoliday"].value_counts()
df["SchoolHoliday"].value_counts()


# In[146]:


plt.figure(figsize=(6,5))
sns.boxplot(x="StateHoliday", y="Sales", data=df)
plt.title("Sales During State Holidays")
plt.show()


# In[147]:


plt.figure(figsize=(6,5))
sns.boxplot(x="SchoolHoliday", y="Sales", data=df)
plt.title("Sales During School Holidays")
plt.show()


# In[148]:


df["StoreType"].value_counts()


# In[149]:


plt.figure(figsize=(6,5))
sns.boxplot(x="StoreType", y="Sales", data=df)
plt.title("Sales by Store Type")
plt.show()


# In[150]:


plt.figure(figsize=(6,5))

sns.scatterplot(
    x="Customers",
    y="Sales",
    data=df
)
plt.title("Customers vs Sales")


# In[151]:


plt.figure(figsize=(6,5))

sns.scatterplot(
    x="CompetitionDistance",
    y="Sales",
    data=df
)
plt.title("Competition Distance vs Sales")


# In[152]:


plt.figure(figsize=(6,5))

sns.boxplot(
    x="DayOfWeek",
    y="Sales",
    data=df
)
plt.title("Sales by Day of Week")


# In[153]:


X = df[[
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Customers",
    "CompetitionDistance",
    "DayOfWeek",
    "Promo2",
    "Month",
    "Year"
]]

y = df["Sales"]


# In[156]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[157]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)


# In[160]:


daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

plt.figure(figsize=(15,5))

plt.plot(
    daily_sales["Date"],
    daily_sales["Sales"],
    color="blue"
)

plt.title("Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.show()


# In[161]:


weekly_sales = df.groupby(
    pd.Grouper(key="Date", freq="W")
)["Sales"].sum().reset_index()

plt.figure(figsize=(15,5))

plt.plot(
    weekly_sales["Date"],
    weekly_sales["Sales"],
    color="green",
    marker="o"
)

plt.title("Weekly Sales")
plt.xlabel("Week")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.show()


# In[162]:


monthly_sales = df.groupby(
    pd.Grouper(key="Date", freq="M")
)["Sales"].sum().reset_index()

plt.figure(figsize=(15,5))

plt.plot(
    monthly_sales["Date"],
    monthly_sales["Sales"],
    color="red",
    marker="o"
)

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.show()


# In[165]:


import matplotlib.pyplot as plt
import pandas as pd

# Daily Sales
daily = df.groupby("Date")["Sales"].sum()

# Weekly Sales
weekly = df.groupby(pd.Grouper(key="Date", freq="W"))["Sales"].sum()

# Monthly Sales
monthly = df.groupby(pd.Grouper(key="Date", freq="ME"))["Sales"].sum()

plt.figure(figsize=(15,6))

plt.plot(daily.index, daily.values, label="Daily Sales", color="blue")

plt.plot(weekly.index, weekly.values, label="Weekly Sales", color="green")

plt.plot(monthly.index, monthly.values, label="Monthly Sales", color="red")

plt.scatter(daily.index, daily.values, color="blue", s=8)

plt.scatter(weekly.index, weekly.values, color="green", s=30)

plt.scatter(monthly.index, monthly.values, color="red", s=60)

plt.title("Daily, Weekly and Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()


# In[ ]:




