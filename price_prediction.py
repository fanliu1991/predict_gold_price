
# coding: utf-8

# In[1]:


# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
# import seaborn

# fix_yahoo_finance is used to fetch data
import fix_yahoo_finance as yf


# In[35]:


# Read data
Df = yf.download('GLD','2008-01-01','2017-12-31')

# Only keep close columns
'''
Df["<column>"]
-- select column <column>, return Series type.

Df[["<column>"]]
-- return a view of columns by passing a list into the __getitem__ syntax (the []'s), return Dataframe type.
'''
close_price = Df[["Close"]]

# Drop rows with missing values
close_price_clean = close_price.dropna(axis=0, how="any")

# Plot the closing price of GLD
close_price_clean.plot(figsize=(10,5))
plt.ylabel("Gold ETF Price")
plt.show()


# In[48]:


# Choose the average of closing price in the past 3 days and 9 days as variables that determines the gold price of next day
'''
.shift(1)
-- copy original closing price, move down 1 unit, add to database

.rolling(window=3).mean()
-- set moving window size = 3 (including itself), this is the number of observations used for calculating the statistic
'''
close_price_clean["Shift"] = close_price_clean["Close"].shift(1)
close_price_clean["S_3"] = close_price_clean["Shift"].rolling(window=3).mean()
close_price_clean["S_9"] = close_price_clean["Shift"].rolling(window=9).mean()

# Drop rows with invalid values
close_price_clean = close_price_clean.dropna()
print close_price_clean.head(10)


# In[49]:


# Define independent variable 
past_average_price = close_price_clean[["S_3", "S_9"]]
print past_average_price.head(10)

# Define dependent variable 
current_price = close_price_clean["Close"]
print current_price.head(10)


# In[61]:


# Split dataset as training dataset and test dataset
t = 0.8
split_t = int(t * len(Df))

# Training dataset
X_train = past_average_price[:split_t]
y_train = current_price[:split_t]

# Test dataset
X_test = past_average_price[split_t:]
y_test = current_price[split_t:]
print X_test.head(10)


# In[58]:


# Fit the linear regression model
regression_model = LinearRegression().fit(X_train, y_train)

# Coefficients and intercept
coefficient = regression_model.coef_
intercept = regression_model.intercept_
print "Gold ETF Price =", coefficient[0], "* 3 Days Moving Average +", coefficient[1], "* 9 Days Moving Average +", intercept


# In[64]:


# Predict gold price in the test dataset
predicted_price = regression_model.predict(X_test)
predicted_price_df = pd.DataFrame(predicted_price, index = y_test.index, columns = ["predicted_price"])
print predicted_price_df.head(10)


# In[65]:


# Plot the predicted closing price of GLD
predicted_price_df.plot(figsize=(10,5))
y_test.plot()
plt.legend(["predicted_price", "actual_price"])
plt.ylabel("Gold ETF Price")
plt.show()


# In[69]:


r2_score = regression_model.score(X_test, y_test)
print r2_score

