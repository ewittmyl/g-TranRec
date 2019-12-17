import gtr
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("data_set/bal_set.csv")
X = df[df.columns[:-4]]
y = df.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=425)

max_features = [2]
min_samples_leaf = [1]

param_grid = {'max_features':max_features,
             'min_samples_leaf':min_samples_leaf}
gtr.RF_GridSearch(param_grid, X_train, y_train)
