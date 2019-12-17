import gTranRec as gtr

max_features = [2,3,4,5,6]
min_samples_leaf = [1,2,3]

param_grid = {'max_features':max_features,
             'min_samples_leaf':min_samples_leaf}
gtr.RF_GridSearch(param_grid, X_train, y_train)
