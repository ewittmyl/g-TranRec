import gTranRec as gtr
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import pkg_resources
import os
from gTranRec import config
from sklearn.ensemble import RandomForestRegressor


#gtr.PrepImages('rb_images.txt', label='both')
#gtr.PrepImages('b_images.txt', label='bogus')
# gtr.PCA_fit(cov_lo=0.01)
#os.system("mkdir -p {}".format(getattr(config, 'MODEL_DIR')))
# os.system("cp pca.m {}".format(getattr(config, 'MODEL_DIR')))
# gtr.AddPCA()
gtr.AddMag2Lim()
gtr.create_df(outfile='all_data.csv')
# gtr.data_sampling(infile='all_data.csv', outfile='bal_set.csv', real2bogus=[400000,400000])

# df = pd.read_csv("bal_set.csv")
# pca = pickle.load(open('pca.m','rb'))
# X_col = ['pca{}'.format(i) for i in range(1,pca.components_.shape[0]+1)] + ['b_image','nmask','n3sig7','gauss_amp','gauss_R','abs_pv','y']
# df = df[X_col]
# X = df[X_col[:-1]]
# y = df.y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=425)
# training_df = X_train.copy()
# training_df['y'] = y_train
# test_df = X_test.copy()
# test_df['y'] = y_test
# training_df.to_csv("train_set.csv", index=False)
# test_df.to_csv("test_set.csv", index=False)

# rf = RandomForestRegressor(n_estimators=150, max_features=5, min_samples_leaf=1,random_state=425, n_jobs=-1, verbose=0)
# X_col = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'b_image', 'nmask', 'n3sig7', 'gauss_amp', 'gauss_R', 'abs_pv']
# rf.fit(training_df[X_col], training_df.y)
# pickle.dump(rf, open('rf.m', 'wb'))

# os.system("cp rf.m {}".format(getattr(config, 'MODEL_DIR')))

