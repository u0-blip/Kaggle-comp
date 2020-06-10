# %%
import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]*10
kf = KFold(n_splits=4)
for train, test in kf.split(X):
    print("%s %s" % (len(train), len(test)))

# %%
