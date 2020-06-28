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
from time import time
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())
df_train = pd.read_csv('data\\train.csv')
df_test = pd.read_csv('data\\test.csv')

X_train = df_train.to_numpy()
X_test = df_test.to_numpy()

y_train = X_train[:, 0]
X_train = X_train[:, 1:]
X_train = X_train.reshape([-1, 28, 28, 1])

# %% Data augmentation
# With data augmentation to prevent overfitting (accuracy 0.99286)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
gen = datagen.flow(X_train, y_train, batch_size=42000)
X_train, y_train = gen[0]
X_train = X_train.reshape([-1, 28*28])

# %% PCA analysis
pca = PCA(n_components=30)
pca.fit(X_train)

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('Explained variance')
# ax1.set_ylabel('explained variance', color=color)
# ax1.plot(pca.explained_variance_ratio_, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Singular values', color=color)  # we already handled the x-label with ax1
# ax2.plot(pca.singular_values_, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()





# %%
X_train = pca.transform(X_train/255)
X_test = pca.transform(X_test/255)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=56)
# model = xgb.XGBclassifier(max_depth=5, n_estimators=100, learning_rate=0.1,objective='reg:squarederror',n_jobs=-1, gpu_id=0)

model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1, gpu_id=0)


prev = time()

model.fit(X_train, y_train)

after = time()

print('train accuracy', model.score(X_train, y_train))
print('testing accuracy', model.score(X_val, y_val))
print('elapsed: ' + str(after - prev))



"""
without PCA 
train accuracy 0.9989100421450371
testing accuracy 0.953211275791921
elapsed: 714.073638677597

with PCA
train accuracy 0.9733928571428572
testing accuracy 0.9401190476190476
elapsed: 139.10289216041565

PCA and normalization
train accuracy 0.9774553571428571
testing accuracy 0.9449404761904762
elapsed: 113.83496117591858

PCA and norm and data aug
train accuracy 0.881875
testing accuracy 0.8280952380952381
elapsed: 121.35249853134155
"""


# %%
