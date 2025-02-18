#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:55:54 2024

@author: ocean
"""

# CRA Project 1, Ocean Yu (my2821), Yiyao Li(yl5567)

import pandas as pd
import numpy as np
import datetime as dt
from sklearn import metrics
import statsmodels.api as sm
import itertools


path = '/Users/dwd4/Documents/Credit Risk Analytics/data/'


findata = pd.read_pickle(path + "bank financials.pickle")
findata = findata[findata.deposits > 0] # removing the trouble data

# Step 1
findata['expense'] = findata['nonIE'] / (findata['TII'] + findata['nonII'])
findata['loan_to_deposit'] = findata['loans'] / findata['deposits']
findata['acl_to_loans'] = findata['allowance'] / findata['loans']

# Standardize the data
factors_to_normalize_assets = ['TII','nonIE', 'nonII', 'deposits', 'staff', 'cash', 'securities', 'equity']
factors_to_normalize_loans = ['TCOs','tot_recoveries', 'allowance']
for factor in factors_to_normalize_assets:
    findata[factor] = findata[factor] / findata['loans']
for factor in factors_to_normalize_loans:
    findata[factor] = findata[factor] / findata['assets']

    
# Step 2
#%%


defaults = pd.read_pickle(path + "default_data.pickle")


data = pd.merge(pd.DataFrame(findata), pd.DataFrame(defaults)[['IDRSSD', 'default', 'dflt_date']], on = 'IDRSSD', how = 'left')

data['default'] = data['default'].fillna(0)

####  DWD Note
#### Your construction doesn't work because it brings in the default flag from the default table which is alway on or off for 
#### Each bank and then you don't turn it off when you set it to one if default occurs within the window.
### you flag banks that default 10 years latter
###  If you replace line 47 above with:
###  data['defautt'] = 0   
### then it will work.
data.loc[(data.dflt_date >= (data.date + dt.timedelta(days=183))) & (data.dflt_date < (data.date + dt.timedelta(days=549))), 'default'] = 1
data.loc[(data.default == 0), 'dflt_date'] = dt.date(2999,12,31)


###DWD Addition you a flagging banks that do not default for ten years
data['default'].describe()
subset = data[data['default']==1]
print( subset[['IDRSSD','date','default','dflt_date']].head(10))



#%%

# Step 3
#DWD coppied before dropping  so I could look at the time series of SVB

data2 = data.copy()
data = data.drop(columns = ['name', 'mutual','date','dflt_date'])

a = data.corr()
a = a['default']
auc_assets = metrics.roc_auc_score(data.default, data.TII)

print('AUC for assets' , auc_assets)

# Step 4
factors = ['expense', 'nonIE', 'loan_to_deposit', 'acl_to_loans', 'TII', 'nonII', 'loans', 'deposits', 'allowance'] 
factor_combinations = list(itertools.combinations(data[factors], 4))
results = []

for combination in factor_combinations:
    x = data[list(combination)]
    x = sm.add_constant(x)
    y = data['default']
    model = sm.Logit(y, x).fit()
    predicted_probs = model.predict(x)
    auc = metrics.roc_auc_score(y, predicted_probs)
    
    # Check for correlation between factors that is higher than 50%
    correlation_matrix = x.corr().abs()
    high_corr = False
    for i in range(len(combination)):
        for j in range(i + 1, len(combination)):
            if correlation_matrix.iloc[i, j] > 0.5:
                high_corr = True
                break
        if high_corr:
            break
    
    if not high_corr:
        results.append((combination, model, auc))

results = sorted(results, key=lambda x: x[2], reverse=True)

print("-------")
# Examine the top three regressions
top_3_results = results[:3]
for i, (factors, model, auc) in enumerate(top_3_results, 1):
    print(f"Model {i}: Factors: {factors}, AUC: {auc}")
    # Calculate pseudo-R-squared using log-likelihood
    ll_null = model.llnull
    ll_model = model.llf
    pseudo_r_squared = 1 - (ll_model / ll_null)
    print(f"Pseudo-R-squared for Model {i}: {pseudo_r_squared}")
# the pseudo_r_squared for these three regressions are not significantly different

# calculate PD
for i, (factors, model, auc) in enumerate(top_3_results, 1):
    X = data[list(factors)]
    X = sm.add_constant(X)
    y = data['default']
    predicted_probs = model.predict(X)
    print(f"Model {i} PD: {predicted_probs}")
    print(f"Model {i} AUC: {auc}")
    
print("-------")

# the AUC for each regression are not significantly different

chosen_model_factors, chosen_model, chosen_model_auc = top_3_results[0]
print(chosen_model.summary())

# Step 5
silicon_valley_id = defaults[defaults['name'] == 'Silicon Valley Bank']['IDRSSD'].values[0]
signature_bank_id = defaults[defaults['name'] == 'Signature Bank']['IDRSSD'].values[0]

silicon_valley_data = data[data['IDRSSD'] == silicon_valley_id]
signature_bank_data = data[data['IDRSSD'] == signature_bank_id]

#DWD Note
#This is a good approach.  Find the IDRSSD associated with the bank by name and than use it going forward.
#The spelling of the name can change (and did in this case). This appraoch is robust to this issue.
#see next three lines that I added.

silicon_valley_data2 = data2[data2['IDRSSD'] == silicon_valley_id]
signature_bank_data2 = data2[data2['IDRSSD'] == signature_bank_id]

print(silicon_valley_data2[['IDRSSD','name','date']])

#%%
X_silicon_valley = silicon_valley_data[list(chosen_model_factors)]
X_silicon_valley = sm.add_constant(X_silicon_valley)
predicted_prob_silicon_valley = chosen_model.predict(X_silicon_valley)

X_signature = signature_bank_data[list(chosen_model_factors)]
X_signature = sm.add_constant(X_signature)
predicted_prob_signature = chosen_model.predict(X_signature)

print(f"Predicted probability of default for Silicon Valley Bank: {predicted_prob_silicon_valley.values}")
print(f"Predicted probability of default for Signature Bank: {predicted_prob_signature.values}")

# the model did not capture the defaults, other factors like Interest Rates, Liquidity Ratios, Capital Adequacy Ratio may have done this better.
    
    
    