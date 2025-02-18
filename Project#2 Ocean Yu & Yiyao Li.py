#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:57:54 2024

@author: ocean
"""

import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True 
import datetime as dt
from statsmodels.tsa.stattools import adfuller
import pandas_datareader.data as web    
import statsmodels.api as sm
from itertools import combinations

# Read the data
card_data = pd.read_excel('/Users/ocean/Desktop/CRA/Data/card.xlsx')
cre_data = pd.read_excel('/Users/ocean/Desktop/CRA/Data/CRE.xlsx')
card_data = pd.DataFrame(card_data)
cre_data = pd.DataFrame(cre_data)

# convert into charge-off percentages
card_data['card_chargeoff_pct'] = card_data.chargeoffs/card_data.loans
cre_data['cre_chargeoff_pct'] = cre_data.chargeoffs/cre_data.loans

# Augmented Dickie Fuller test
adfuller(card_data.card_chargeoff_pct)
adfuller(cre_data.cre_chargeoff_pct)
print(f'\nThe ADFuller is {100*adfuller(card_data.card_chargeoff_pct)[1]:3.3f}%')
print(f'\nThe ADFuller is {100*adfuller(cre_data.cre_chargeoff_pct)[1]:3.3f}%')
""" card data test: The p-value for the ADFuller test for card data is 5.327%, which failed to reject the null hypothesis under 5% significant value, suggesting that the data is borderline stationary. Therefore, we just treat it as stationary """
""" cre data test: The p-value for the ADFuller test for cre data is 48.913%, which failed to reject the null hypothesis, suggesting that the data is non-stationary."""

# Take the first difference to make the series stationary
# card_data['chargeoff_pct'] = card_data['chargeoff_pct'].diff() --> disregard this since we just treat it as stationary
cre_data['cre_chargeoff_pct'] = cre_data['cre_chargeoff_pct'].diff()

# using isna() to skip the NA value
adfuller(cre_data['cre_chargeoff_pct'][~cre_data['cre_chargeoff_pct'].isna()])
print(f'\nThe ADFuller is {100*adfuller(cre_data.cre_chargeoff_pct[~cre_data.cre_chargeoff_pct.isna()])[1]:3.3f}%')
"""Based on the ADFuller test, the first difference cre series is now stationary since the p-value is lower than 5%."""

# Merge the cre and card data
merged_data = pd.merge(card_data[['date', 'card_chargeoff_pct']], cre_data[['date','cre_chargeoff_pct']], on=['date'])

# Download data
unrate = web.DataReader("UNRATE", "fred", start = '2000-01-01')
oil_price = web.DataReader('DCOILBRENTEU', "fred", start = '2000-01-01')
gdp = web.DataReader("GDP", "fred", start = '2000-01-01')
t10y2y = web.DataReader('T10Y2Y', 'fred', start = '2000-01-01')
vix = web.DataReader('VIXCLS', 'fred', start = '2000-01-01')

unrate['date'] = [x.date() - dt.timedelta(days=1) for x in unrate.index]
unrate = unrate[unrate['date'] <= dt.date(2020, 3, 1)]
unrate['year'] = [x.year for x in unrate.date]
unrate['month'] = [x.month for x in unrate.date]
unrate = unrate[~unrate.UNRATE.isna()]
unrate = unrate[unrate.month.isin([3,6,9,12])]
adfuller(unrate.UNRATE)
print(f'\nThe ADFuller for unrate is {100*adfuller(unrate.UNRATE)[1]:3.3f}%')
"""Based on the ADFuller test, the unrate series is non stationary, let's try taking the quarterly differences"""
unrate['UNRATE'] = unrate['UNRATE'].diff()
unrate = unrate[~unrate.UNRATE.isna()]
adfuller(unrate.UNRATE)
print(f'\nThe ADFuller for the new unrate is {100*adfuller(unrate.UNRATE)[1]:3.3f}%')
"""Based on the ADFuller test result, the unrate series after taking the first difference is now stationary"""

# Let's will do the same process for all series
oil_price['date'] = [x.date() - dt.timedelta(days=1) for x in oil_price.index]
oil_price = oil_price[oil_price['date'] <= dt.date(2020, 3, 1)]
oil_price['year'] = [x.year for x in oil_price.date]
oil_price['month'] = [x.month for x in oil_price.date]
oil_price['day'] = [x.day for x in oil_price.date]
oil_price = oil_price.resample('M').last() # make sure only 1 observarion each month
oil_price = oil_price[~oil_price.DCOILBRENTEU.isna()]
oil_price = oil_price[oil_price.month.isin([3,6,9,12])]
adfuller(oil_price.DCOILBRENTEU)
print(f'\nThe ADFuller for oil_price is {100*adfuller(oil_price.DCOILBRENTEU)[1]:3.3f}%')
oil_price['oil_price_lag'] = oil_price.DCOILBRENTEU.shift()
oil_price['oil_price_growth'] = (oil_price.DCOILBRENTEU - oil_price.oil_price_lag)/oil_price.oil_price_lag
oil_price = oil_price[~oil_price.oil_price_growth.isna()]
adfuller(oil_price.oil_price_growth)
print(f'\nThe ADFuller for the oil_price_growth is {100*adfuller(oil_price.oil_price_growth)[1]:3.3f}%')
"""Based on the ADFuller test, the oil_price_growth is now stationary"""


t10y2y['date'] = [x.date() - dt.timedelta(days=1) for x in t10y2y.index]
t10y2y = t10y2y[t10y2y['date'] <= dt.date(2020, 3, 1)]
t10y2y['year'] = [x.year for x in t10y2y.date]
t10y2y['month'] = [x.month for x in t10y2y.date]
t10y2y = t10y2y.resample('M').last() # make sure only 1 observarion each month
t10y2y = t10y2y[~t10y2y.T10Y2Y.isna()]
t10y2y = t10y2y[t10y2y.month.isin([3,6,9,12])]
adfuller(t10y2y.T10Y2Y)
print(f'\nThe ADFuller for t10y2y is {100*adfuller(t10y2y.T10Y2Y)[1]:3.3f}%')
t10y2y['T10Y2Y'] = t10y2y['T10Y2Y'].diff()
t10y2y = t10y2y[~t10y2y.T10Y2Y.isna()]
adfuller(t10y2y.T10Y2Y)
print(f'\nThe ADFuller for the new t10y2y is {100*adfuller(t10y2y.T10Y2Y)[1]:3.3f}%')
"""Based on the ADFuller test, the new t10y2y is now stationary"""

vix['date'] = [x.date() - dt.timedelta(days=1) for x in vix.index]
vix = vix[vix['date'] <= dt.date(2020, 3, 1)]
vix['year'] = [x.year for x in vix.date]
vix['month'] = [x.month for x in vix.date]
vix = vix.resample('M').last().dropna() # make sure only 1 observarion each month
vix = vix[~vix.VIXCLS.isna()]
vix = vix[vix.month.isin([3,6,9,12])]
adfuller(vix.VIXCLS)
print(f'\nThe ADFuller for vix is {100*adfuller(vix.VIXCLS)[1]:3.3f}%')
""" This series is not stationary """
vix['VIXCLS'] = vix['VIXCLS'].diff()
vix = vix[~vix.VIXCLS.isna()]
adfuller(vix.VIXCLS)
print(f'\nThe ADFuller for the new vix is {100*adfuller(vix.VIXCLS)[1]:3.3f}%')
""" Now it is stationary """


# Pull the data in to a pandas df
economic_data = pd.merge(unrate, oil_price[['oil_price_growth','month','year']], how='left', on = ['month', 'year'])
economic_data = pd.merge(economic_data, t10y2y[['T10Y2Y','month','year']], how='left', on = ['month', 'year'])
economic_data = pd.merge(economic_data, vix[['VIXCLS','month','year']], how='left', on = ['month', 'year'])

# Create GDP growth variable
gdp['date'] = [x.date() - dt.timedelta(days=1) for x in gdp.index]
gdp = gdp[gdp.date <= dt.date(2020, 3, 1)]
gdp['month'] = [x.month for x in gdp.date]
gdp['year'] = [x.year for x in gdp.date]
gdp['gdp_lag'] = gdp.GDP.shift()
gdp['gdp_growth'] = (gdp.GDP - gdp.gdp_lag)/gdp.gdp_lag
gdp = gdp[~gdp.gdp_growth.isna()]
adfuller(gdp.gdp_growth)
print(f'\nThe ADFuller for gdp_growth is {100*adfuller(gdp.gdp_growth)[1]:3.3f}%')
""" This series is stationary"""

# Merge the gdp data into our economic_data
economic_data = pd.merge(economic_data, gdp[['gdp_growth','month','year']], how='left', on = ['month', 'year'])

# Combine data with charge-off data
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['year'] = [x.year for x in merged_data.date]
merged_data['month'] = [x.month for x in merged_data.date]
merged_data = pd.merge(merged_data, economic_data, how='left', on=['month', 'year'])

# Creating lag data for both chargeoffs
merged_data['card_chargeoff_pct_lag'] = merged_data['card_chargeoff_pct'].shift()
merged_data['cre_chargeoff_pct_lag'] = merged_data['cre_chargeoff_pct'].shift()

# Run all possible AR1, three-factor models
best_card_r_squared = -np.inf
best_cre_r_squared = -np.inf
best_card_model = None
best_cre_model = None
best_card_factors = None
best_cre_factors = None

# for card data
factors = ['UNRATE', 'oil_price_growth', 'T10Y2Y', 'VIXCLS', 'gdp_growth']
for comb in combinations(factors, 3):
    X_card = list(comb) + ['card_chargeoff_pct_lag']
    Y_card = 'card_chargeoff_pct'
    card_model_data = merged_data[[Y_card] + X_card]
    X_card_data = card_model_data[X_card]
    # Since there are NA values in chargeoff lag data, we should skip it using isna()
    X_card_data = X_card_data[~X_card_data.card_chargeoff_pct_lag.isna()]
    y_card_data = card_model_data[Y_card][1:] # since we skipped the first raw, we should drop the first y as well to match the data size
    X_card_data = sm.add_constant(X_card_data)
    card_model = sm.OLS(y_card_data, X_card_data)
    card_results = card_model.fit()
    if card_results.rsquared > best_card_r_squared:
        best_card_r_squared = card_results.rsquared
        best_card_model = card_results
        best_card_factors = comb
print(best_card_factors, best_card_r_squared)
# for cre data
for comb in combinations(factors, 3):
    X_cre = list(comb) + ['cre_chargeoff_pct_lag']
    Y_cre = 'cre_chargeoff_pct'
    cre_model_data = merged_data[[Y_cre] + X_cre]
    X_cre_data = cre_model_data[X_cre]
    # Since there are NA values in chargeoff lag data, we should skip it using isna()
    X_cre_data = X_cre_data[~X_cre_data.cre_chargeoff_pct_lag.isna()]
    y_cre_data = cre_model_data[Y_cre][2:] # since the first 2 raws are NA, we skipped the first two raw, we should drop the first y as well to match the data size
    X_cre_data = sm.add_constant(X_cre_data)
    cre_model = sm.OLS(y_cre_data, X_cre_data)
    cre_results = cre_model.fit()
    if cre_results.rsquared > best_cre_r_squared:
        best_cre_r_squared = cre_results.rsquared
        best_cre_model = cre_results
        best_cre_factors = comb
print(best_cre_factors, best_cre_r_squared)

""" 
Results: 
    Card:('UNRATE', 'oil_price_growth', 'gdp_growth') 0.8667314353677347
    CRE: ('UNRATE', 'VIXCLS', 'gdp_growth') 0.3977107392249346
    
    Comments:
    The Card Chargeoffs Model has a high R-squared value, meaning that approximately 86.88% of the variance in the card chargeoff percentage can be explained by the factors 
    UNRATE (Unemployment Rate), oil_price_growth (Oil Price Growth), and VIXCLS (Volatility Index). This is a high R-squared value, indicating a strong fit of the model to the data.
   
    The CRE Chargeoffs Model has a lower R-squared value, meaning that approximately 33.72% of the variance in the CRE chargeoff percentage can be explained by the factors 
    UNRATE (Unemployment Rate), T10Y2Y (10-Year Minus 2-Year Treasury Yield Spread), and gdp_growth (GDP Growth). This is a moderate R-squared value, indicating a weaker fit of the model to the data compared to the card chargeoffs model.
    This suggests that additional or alternative factors may be needed to improve the model's predictive power for CRE chargeoffs.
    
    The reason behind difference in R-squared is that for the CRE data, we are predicting change in CRE charges offs, but in CARDS we are predicting the levels of Credit Offs 
    because we took the first difference of the CRE data. 
""" 

""" 
Step 3: 
    Other factors:
        Interest rates, mortgage rates, Inflation rates, credit spread, and bank-specifit capital ratios could all be useful factors in this exercise.
        We need to make sure the stationarity for these data when using them in our model.
        
    Information needed to make forecasts:
        First, we would need the updated data on the factors used in the model.
        Second, we would need a comprehensive historical dataset to re-estimate models periodically and ensure they remain accurate and relevant.
        Third, we would need a real-time or near-real-time data to ensure forecasts are based on the most current information.
        We might also need information on any changes in banking regulations or economic policies that could impact chargeoff rates.
    
    If we are talking about things to be awared of when using AR model:
        We need to make sure that the data is time series and there should be a lagged factor.
        We need to make sure that seriers are all stationary.
"""
    
    
    
    
    
    
    
    
    
    
    
    