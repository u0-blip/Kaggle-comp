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

sys.path.append(os.getcwd())

df_train = pd.read_csv('data\\train.csv')
df_test = pd.read_csv('data\\test.csv')

dropped = None

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
