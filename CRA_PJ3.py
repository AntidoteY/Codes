#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:29:19 2024

@author: leoli
"""

import warnings
import numpy as np
warnings.filterwarnings("ignore")

''' 
Step 1:
The first step is to create a portfolio of 1,000 borrowers. 
Each borrower should have a long-term, fundamental, annualized PD drawn from a random, 
uniform distribution between 0 and 5 percent. 
'''
borrowers = 1000  
pd_min = 0.0  
pd_max = 0.05  

lt_pds = np.random.uniform(pd_min, pd_max, borrowers)

# Convert to percentage and round to 4 decimal places
lt_pds = [round(pds, 4) for pds in lt_pds]
print('first 10 annual pds:')
print(lt_pds[:10])


'''
Step 2:
We intend to simulate a five-year history based on macroeconomic variables. 
We’ll keep it simple with one variable, the unemployment rate. The equation we will use will be:
UR_t = 0.01 + 0.8 UR_t-1 + epsilon_t-1
Even though unemployment is updated every month, we’ll pretend it is updated quarterly. 
UR at time zero will be assumed to be 3.6%. 
The residual term is assumed to have a normal distribution with a quarterly standard deviation of 0.25%. 
We will assume that, no matter what, the UR at the end of the 5th simulation quarter and the 13th simulation quarter is 10%.
Create a list with 20 entries representing the UR rate at the end of each quarter.'''

UR0 = 0.036
quarters_UR = 20
stdv_UR = 0.0025
eps_UR = 0.01*np.random.normal(0, stdv_UR, quarters_UR)
UR = [UR0]

for t in range(1, quarters_UR):
    UR_t = 0.01 + 0.8*UR[-1] + eps_UR[t-1]
    UR.append(UR_t)

UR[4] = 0.1
UR[12] = 0.1

# Convert to percentage and round to 4 decimal places
UR = [round(rate, 4) for rate in UR]
print(UR)
print('-----'*10)
''' 
Step 3:
We will use these simulations to create a PD multiple. 
We could estimate the relationship empirically but we are trying to keep things easy for now. 
We’ll just divide the list from Step 2 with the average of the list entries. 
This is nice because it guarantees that the overall average is 1.0 even if the actual number varies over time.
For each time period multiple each firms PD by the multiplier to get the adjusted PD. 
'''
# get PD multipliers
average_UR = np.mean(UR)
multipliers = [ur / average_UR for ur in UR]

# Apply multipliers to each long-term PD to get the adjusted PDs
adjusted_PDs = []
for multiplier in multipliers:
    adjusted_PD = [round(pds * multiplier, 4) for pds in lt_pds]
    adjusted_PDs.append(adjusted_PD)
    

''' 
Step 4:
Convert each default rate into a quarterly default rate using the following equation:
PD_quarterly = 1 - (1 - PD)^(1/4)
This can also be done approximately simply by dividing by 4.
'''
PDs_quarterly = []
for adjusted_PD_list in adjusted_PDs:
    pd_quarterly = [round(1 - (1 - pds)**(1/4), 4) for pds in adjusted_PD_list]
    PDs_quarterly.append(pd_quarterly)

'''
Step 5:
Simulate the default history, by walking through quarter by quarter, 
rolling one uniform random # for each surviving customer and 
seeing if it is less than the quarterly PD times the multiplier from Step 3. 
If it is, mark that customer as defaulted.
The best practice here would be to keep track of 
how many customers exist at the beginning of each quarter as well as how many defaulted during the quarter.
'''

num_customers_beg = []
num_defaults = []

# Initialize the list of surviving customers (initially all customers)
surviving_customers = np.ones(borrowers, dtype=bool)

for quarter in range(quarters_UR):
    # Count the number of surviving customers at the beginning of the quarter
    num_surviving = np.sum(surviving_customers)
    num_customers_beg.append(num_surviving)
    
    # Calculate defaults for this quarter
    defaults_this_quarter = 0
    for i in range(borrowers):
        if surviving_customers[i]:  # Only consider non-defaulted customers
            random_number = np.random.uniform(0, 1)
            if random_number < (PDs_quarterly[quarter][i] * multipliers[quarter]):
                surviving_customers[i] = False
                defaults_this_quarter += 1
    
    num_defaults.append(defaults_this_quarter)

# Display the number of customers at the beginning of each quarter and the number of defaults
print("Number of customers at the beginning of each quarter:",  num_customers_beg)
print("Number of defaults each quarter:", num_defaults)
print('-----'*10)

'''
Step6:
Calculate the historical default rate in annual terms, 
by (1) the pooled method where you add up all the counts and all the defaults and divide 
or (2) calculating the default rate for each quarter and then averaging all 20 quarterly PDs. 
In either case, you will need to reverse the process in Step 4 to get an annualized PD.
'''
# Method 1: Pooled Method
total_defaults = sum(num_defaults)
total_customer_quarters = sum(num_customers_beg)
pooled_quarterly_PD = total_defaults / total_customer_quarters
annualized_PD_pooled = 1 - (1 - pooled_quarterly_PD)**4
print("Pooled Quarterly PD:", pooled_quarterly_PD)
print("Pooled Annualized PD:", annualized_PD_pooled)
print('-----'*10)
# Method 2: Average Quarterly Method
quarterly_default_rates = [num_defaults[i] / num_customers_beg[i] for i in range(quarters_UR)]
average_quarterly_PD = np.mean(quarterly_default_rates)
annualized_PD_average = 1 - (1 - average_quarterly_PD)**4
print("Average Quarterly PD:", average_quarterly_PD)
print("Average Annualized PD:", annualized_PD_average)



