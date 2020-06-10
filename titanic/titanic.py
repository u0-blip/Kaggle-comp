# %% # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
sys.path.append(os.getcwd())
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
# train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
# test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


train_data = pd.read_csv(".\\titanic\\train.csv")
test_data = pd.read_csv('.\\titanic\\test.csv')


# %% # Any results you write to the current directory are saved as output.
h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# %%


from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

X_out = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=1)

model.fit(X_train, y_train)
predictions = model.predict(X_out)

print('train accuracy: ', model.score(X_train, y_train))
print('test accuracy: ', model.score(X_test, y_test))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


# %%
# iterate over classifiers
d = {'names': names, 'train_score': np.zeros((len(names))), 'test_score': np.zeros((len(names)))}

scores = pd.DataFrame(d)
train_score = []
test_score = []
for i, name, clf in enumerate(zip(names, classifiers)):

    clf.fit(X_train, y_train)
    train_score.append(clf.score(X_train, y_train))
    test_score.append(clf.score(X_test, y_test))
    scores[i, 'train_score'] = train_score[i]
    scores[i, 'test_score'] = test_score[i]
    
    # print(name, 'train accuracy: ', clf.score(X_train, y_train))
    # print(name, 'test accuracy: ', clf.score(X_test, y_test))
ax = scores.plot.bar(x='names', y='train_score', rot=40)
ax = scores.plot.bar(x='names', y='test_score', rot=40)

# %%
