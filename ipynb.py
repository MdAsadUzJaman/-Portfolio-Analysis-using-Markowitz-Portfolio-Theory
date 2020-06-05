#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing numpy, matplotlib, pandas,pandas_datareader
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Markowitz Portfolio Theory (MPT)
# Modern portfolio theory or Markowitz Portfolio Theory (MPT) is a theory that helps construct portfolios to optimize or maximize expected return based on a given level of market risk for risk averse investors, keeping that in mind that risk is inevitable part of return. Risk and return are positively correlated. However, the theory paves the way to construct an "efficient frontier" of optimal portfolios that generate the highest possible expected return for a given level of risk. This concept was introduced by Harry Markowitz in his paper "Portfolio Selection," published in 1952 by the Journal of Finance He was later awarded a Nobel prize for developing the MPT.(Cheng, 2003)
# 
# In finance world, “FAANG” refers to the stocks of five giant American technology companies: Facebook (FB), Amazon (AMZN), Apple (AAPL), Netflix (NFLX); and Alphabet (GOOG). In addition, the five FAANG stocks are among the largest companies in the world, with a combined market capitalization of over $4.1 trillion as of January 2020.(Fernando, 2017)
# 
# We will implement Markowitz Portfolio Theory to construct  an efficient frontier of optimal portfolios.

# In[2]:


# Using ticker symbols of FAANG to download and create DataFrame from 2012-05-18
assets=['FB','AMZN','AAPL','NFLX','GOOG']
pf_data =pd.DataFrame()
for t in assets:
    pf_data[t]=wb.DataReader(t, data_source="yahoo", start="2012-05-18")["Adj Close"]


# In[ ]:


# Showing top 6 rows of data
pf_data.head(6)


# In[ ]:


# Locating missing values
pf_data.info()


# The normalized value for each stock after the base date/time is the percent of the base price expressed as a whole number. Here, 100 times actual price (any given date) divided by actual base price (initail date). We indiate initail date with iloc[0]. This indicator shows the percentage move in price relative to some fixed starting point ("Normalized price" 2015).
# 
# To illustrate, if we want to calculate the normalized price of Facebook("FB") for 5th day, we will devide 5th day (2012-05-18) price which is 33.029999 by base price (2012-05-24) of 38.230000 times 100.

# In[ ]:


# Normalizing stock prices
normalized_price = pf_data/pf_data.iloc[0]*100
normalized_price.plot(figsize=(13,5))


# ### Simple Returns
# The simple return of a portfolio is the weighted sum of the simple returns of the constituents of the portfolio. If simple returns are used than the portfolio return is the weighted average of assets in that portfolio. So one of the advantages of simple return is that it can be used where portfolios are formed and portfolio returns have to be calculated because of its asset-additive property.
# 
# $ Return = (Ending Price - Starting Price) / Starting Price$
# 
# Please note that, we will use simple returns for calculating FAANG portfolio.
# 
# ### Log Returns
# The log return for a time period is the sum of the log returns of partitions of the time period. For example, the log return for a year is the sum of the log returns of the days within the year. Log returns are time-additive, not asset-additive. The weighted average of log returns of individual stocks is not equal to the portfolio return. In fact, log returns are not a linear function of asset weights.
# 
# $ Return = log(Ending Price - Starting Price)$
# 

# In[ ]:


# Calculating log returns just for the record
log_returns=np.log(pf_data/pf_data.shift(1))
log_returns.tail()


# In[ ]:


# Calculating simple returns of FAANG portfolio
simple_returns = (pf_data / pf_data.shift(1)) - 1
simple_returns.head()


# In[ ]:


# Calculating annual simple return from daily simple returns of FAANG portfolio
annual_return = simple_returns.mean()*250
annual_return


# In[ ]:


# Calculating annual covariance from daily simple returns of FAANG portfolio
simple_returns.cov()*250


# In[ ]:


# Calculating correlation among stocks in FAANG portfolio
simple_returns.corr()


# In[ ]:


num_assets = len(assets)
num_assets


# In[ ]:


# Generationg random weights as per number of assets in FAANG Portfolio
weights = np.random.random(num_assets)
weights= weights / np.sum(weights)
# or weights /= np.sum(weights)
weights


# In[ ]:


# Adding random weights to check if equal to 1
weights[0]+weights[1]+weights[2]+weights[3]+weights[4]


# ### Expected Portfolio Return
# Expected portfolio return is the weighted average of the expected return of each of its stocks. The basic expected return formula involves multiplying each asset's weight in the portfolio by its expected return, then adding all those figures together.
# 
# $ Expected  Portfolio  Return = \sum_{i=1}^{n}w_i*r_i $

# In[ ]:


# Calcualting annual expected returns
annual_pfolio_retrun = np.sum(weights*simple_returns.mean())*250
annual_pfolio_retrun


# ### Expected Portfolio Variance
# Portfolio variance is a measurement of risk, of how the aggregate actual returns of a set of securities making up a portfolio fluctuate over time. This portfolio variance statistic is calculated using the standard deviations of each security in the portfolio as well as the correlations of each security pair in the portfolio. Our target is to minimize expeted portfolio variance while maximizing portfolio returns.
# 
# To calculate the portfolio variance of securities in a portfolio, multiply the squared weight of each security by the corresponding variance of the security and add two multiplied by the weighted average of the securities multiplied by the covariance between the securities (Nickolas, 2015). 
# 
# $ Expected Portfolio Variance = \sum_{i=1}^{n}(w_i^2)Var(r_i) + 2\sum_{i=1}^{n}\sum_{j=1}^{n}(w_i)(w_j)Cov(r_i,r_j) $
# 
# We will use following notation for matrix calculation, $ (w.V)^2 = w^TV.w$
# 
# Something like,
# 
# $ (w.V)^2 = [w_1 \dots w_n]
# \begin{bmatrix} 
# \sigma_{1}^2 & \sigma_{1,2} \dots \sigma_{1,n}\\
# \sigma_{2,1} & \sigma_{2}^2 \dots \sigma_{2,n}\\
# \vdots&\vdots\ddots\\
# \sigma_{n,1} & \sigma_{n,2} \dots \sigma_{n}^2\\
# \end{bmatrix} \begin{bmatrix} w_1 \\\vdots\\ w_n\end{bmatrix} $
# 
# 

# In[ ]:


# Calculating annual portfolio variance
annual_pfolio_variance = np.dot(weights.T, np.dot(simple_returns.cov()*250,weights))
annual_pfolio_variance


# In[ ]:


# Calculating annual portfolio standard deviation or volatility
annual_pfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(simple_returns.cov()*250,weights)))
annual_pfolio_volatility


# In[ ]:


# creating empty arrays
pfolio_returns = []
pfolio_volatilities = []

# Creating loop for appending  portfolio returns and volatilities in arrays
for x in range (1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * simple_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(simple_returns.cov() * 250, weights))))
   
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)


# In[ ]:


# Converting portfolio returns and portfolio volatilities arrays to a DataFrame
portfolios = pd.DataFrame({'Return':pfolio_returns, 'Volatility':pfolio_volatilities})


# In[ ]:


portfolios.tail()


# ### Efficient Frontier
# The efficient frontier is the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk (Ganti, 2003).

# In[ ]:


# Ploting Efficient Frontier of FAANG portfolio
portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6));
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier of FAANG Portfolio')
plt.show()


# ### Sharpe Ratio
# The Sharpe ratio was developed by Nobel laureate William F. Sharpe and is used to help investors understand the return of an investment compared to its risk. The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk (Hargrave, 2003).
# 
# $$ Sharpe Ratio = \frac{r_p - r_f}{ \sigma_p}$$

# In[ ]:


# Calculation Sharpe Ratio
# Considering risk free rate (10 Year Treasury Rate of June 4, 2020) is 0.82%. 
# For more info, refer to 'https://ycharts.com/indicators/10_year_treasury_rate'
risk_free_rate = 0.0082
sharpe_ratio = (annual_pfolio_retrun - risk_free_rate)/annual_pfolio_volatility
sharpe_ratio


# References
# 10 year treasury rate. (n.d.). Retrieved from https://ycharts.com/indicators/10_year_treasury_rate
# 
# Cheng, J. (2003, November 24). Modern portfolio theory (MPT). Retrieved from https://www.investopedia.com/terms/m/modernportfoliotheory.asp
# 
# Fernando, J. (2017, June 12). FAANG stock definition. Retrieved from https://www.investopedia.com/terms/f/faang-stocks.asp
# 
# Ganti, A. (2003, November 18). Efficient frontier definition. Retrieved from https://www.investopedia.com/terms/e/efficientfrontier.asp
# 
# Hargrave, M. (2003, November 26). How to use the Sharpe ratio to analyze portfolio risk and return. Retrieved from https://www.investopedia.com/terms/s/sharperatio.asp
# 
# Nickolas, S. (2015, July 15). How can I measure portfolio variance? Retrieved from https://www.investopedia.com/ask/answers/071515/how-can-i-measure-portfolio-variance.asp
# 
# Normalized price. (2015, April 22). Retrieved from https://www.linnsoft.com/techind/normalized-price
