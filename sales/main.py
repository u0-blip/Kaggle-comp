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

sys.path.append(os.getcwd())

df_train = pd.read_csv('data\\train.csv')
# df_train = df_train.iloc[0:int(len(df_train)/10)]
df_test = pd.read_csv('data\\test.csv')

# %%
# shop 50 sales trend
records = {}
for shop in df_train.shop_id.unique():
    record = {}
    for item in df_train.item_id.unique():
        record[item] = df_train[np.logical_and(df_train.shop_id == shop, df_train.item_id == item)][['date', 'item_cnt_day']].groupby('date').sum()
    records[shop] = record


# %%
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    

    #Perform Dickey-Fuller test:
    print( 'Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(shop_50)


# %%
import statsmodels.api as sm
def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    
    # if not isinstance(y, pd.Series):
    #     y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = statsmodels.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        statsmodels.graphics.tsaplots.plot_acf(y, lags=lags, ax=acf_ax)
        statsmodels.graphics.tsaplots.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
tsplot(shop_50, lags=30)

# Take the first difference to remove to make the process stationary
data_diff = shop_50 - shop_50.shift(1)

tsplot(data_diff[1:], lags=30)

# %%
def process_data(df_train, test = False):
    #missing data
    global dropped

    if not test:
        total = df_train.isnull().sum().sort_values(ascending=False)
        percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data.head(20)
        #dealing with missing data
        dropped = (missing_data[missing_data['Total'] > 1]).index


    df_train = df_train.drop(dropped,1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    df_train.isnull().sum().max() #just checking that there's no missing data missing...
    #deleting points
    df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

    if not test:
        df_train['SalePrice'] = np.log(df_train['SalePrice'])

    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    #if area>0 it gets 1, for area==0 it gets 0
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0 
    df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

    df_train = pd.get_dummies(df_train)
    return df_train

X_train = process_data(df_train)
y_train = X_train[['SalePrice']]
X_train = X_train.drop(columns=['SalePrice'])
X_val = process_data(df_test, True)
X_train = X_train.drop(columns = [ele for ele in X_train.columns if (ele not in X_val.columns)])

# %%
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=56)
model = xgb.XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.1,objective='reg:squarederror',n_jobs=-1, gpu_id=0)


prev = time()

model.fit(X_train, y_train)

print('train accuracy', model.score(X_train, y_train))
print('testing accuracy', model.score(X_test, y_test))

after = time()
print('elapsed: ' + str(after - prev))


# %%
price = pd.DataFrame(np.power(np.e, model.predict(X_val)))
price.insert(0, column = 'id', value=X_val['Id'])
price.columns = ['id', 'SalePrice']
# %%
price.to_csv('price.csv', index=False)

# %%
