# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import datetime
import os
import sys
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statsmodels
from scipy.sparse import csr_matrix
import statsmodels.api as sm

sys.path.append(os.getcwd())

data = pd.read_csv('data.csv')


# %%
vic_case = data[['Date', 'VIC', 'NSW', 'QLD', 'SA', 'VIC_Deaths', 'VIC_Tests']]
data_filled = vic_case.fillna(0)
data_filled['new_case'] = vic_case['VIC'] - vic_case['VIC'].shift(1)
data_filled.fillna(0, inplace=True)
data_filled.set_index('Date', inplace=True)
data_filled.index = pd.DatetimeIndex(data_filled.index).to_period('D')
data_filled.sort_values(by='Date', inplace=True)

y = data_filled[['new_case']]

# %%
# Graph data
# fig, axes = plt.subplots(1, 2, figsize=(15,4))

# fig = sm.graphics.tsa.plot_acf(data_filled.iloc[1:]['new_case'], lags=40, ax=axes[0])
# fig = sm.graphics.tsa.plot_pacf(data_filled.iloc[1:]['new_case'], lags=40, ax=axes[1])

# %% testing parameters
# import itertools

# p = d = q = range(4, 6)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

# %%
mod = sm.tsa.statespace.SARIMAX(endog = y,
                                order=(6, 4, 5),
                                seasonal_order=(2,2,2, 7),
                                enforce_stationarity=True,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])

# %% diagnosis
# pd.plotting.register_matplotlib_converters()
# results.plot_diagnostics(figsize=(15, 4))
# plt.show()




# %%
plt.show()
exog = data_filled[['VIC_Tests']][-7:]
pred = results.get_prediction(start=pd.to_datetime('2020-03-22'), end=pd.to_datetime('2020-07-23'), dynamic=False)
pred_ci = pred.conf_int()
ax = y.plot(label='observed', style ='+')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0]*(pred_ci.iloc[:,0]>0),
                pred_ci.iloc[:, 1], color='r', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('New cases')
plt.legend()
plt.show()

mse = sum((y['new_case'][-len(pred.predicted_mean):] - pred.predicted_mean).fillna(0)**2)
print('prediction', pred.predicted_mean[-5:], 'MSE', mse)
# %%


# %%
