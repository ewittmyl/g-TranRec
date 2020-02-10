import os
from astropy.io.fits import getheader, getdata
from .image_process import fits2df, FitsOp
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from . import config



def PCA_fit(cov_lo=0.01):
    """
    This function is to perform PCA transformation on the original feature space. Only the PCA with covariance higher than
    the given 'cov_lo' value. 
    
    The following file would be created:
    1. pca.m

    Parameters
    ----------
    cov_lo: float
        Covariance lower limit for PCA components.
    """
    images = [img for img in os.listdir(".") if '.fits' in img]
    p_col = ['p'+str(i+1) for i in np.arange(441)]
    df = pd.DataFrame(columns=p_col)
    for img in images:
        try:
            print("Appending {}[IMAGE_DETAB] into PCA fitting data...".format(img))
            real_X = fits2df(img, 'IMAGE_DETAB')
            # read-in stamps for the detections on 'IMAGE' 
            real_stamps = fits2df(img, 'IMAGE_STAMPS')
            real_stamps.columns = p_col
            # masking bad sample of the real detections
            real_mask = (real_X.X_IMAGE > 21) & (real_X.X_IMAGE < 8155) & (real_X.Y_IMAGE > 21) & (real_X.Y_IMAGE < 6111) & (real_X.FLAGS == 0) & (real_X.FLAGS_WIN == 0) & (np.abs(real_X.ERRCXYWIN_IMAGE)<100)
            df = df.append(real_stamps[real_mask], ignore_index = True)
        except:
            print("'IMAGE_DETAB' does not exist in {}...".format(img))

        try:
            print("Appending {}[DIFFERENCE_DETAB] into PCA fitting data...".format(img))
            bogus_X = fits2df(img, 'DIFFERENCE_DETAB')
            # read-in stamps for the detections on 'DIFFERENCE' 
            bogus_stamps = fits2df(img, 'DIFFERENCE_STAMPS')
            bogus_stamps.columns = p_col
            # masking bad sample of the bogus detections
            df = df.append(bogus_stamps, ignore_index = True)
        except:
            print("'DIFFERENCE_DETAB' does not exist in {}...".format(img))
    
    # define date set for fitting PCA
    pca_X = df[p_col]

    # PCA fitting and transform X
    cov = 1
    n = 1
    print("Searching for all PCA components with covariance above {}...".format(cov_lo))
    while cov > cov_lo:
        pca = PCA(n_components=n)
        pca.fit(pca_X)
        cov = pca.explained_variance_ratio_[-1]
        n+=1

    pickle.dump(pca, open('pca.m', 'wb'))   

def X_validation(model, X_train, y_train, cv=5):

    X_col = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'b_image',
            'nmask', 'n3sig7', 'gauss_amp', 'gauss_R', 'abs_pv']
    # make sure the training set has the correct format
    X_train = X_train[X_col]
    # re-combine X and y
    X_train['y'] = y_train
    # backup dataframe
    df = X_train.copy()
    # re-ordering
    df = shuffle(df)
    # equally split
    cv_set = np.array_split(df, cv)
    # initialize accuracies
    train_acc, val_acc = [], []
    for i in range(5):
        # define test and train for CV
        test = pd.DataFrame(cv_set[i])
        train = pd.DataFrame(np.vstack([tab for k, tab in enumerate(cv_set) if k!=i]))
        X_train = train[train.columns[:-1]]
        y_train = train[train.columns[-1]]
        X_test = test[test.columns[:-1]]
        y_test = test[test.columns[-1]]
        # fit on train
        model.fit(X_train, y_train)
        # calculate training acc
        train_pred = model.predict(X_train)
        train_pred = [int(round(s)) for s in train_pred]
        trac = np.sum(y_train == train_pred)/y_train.shape[0]
        train_acc.append(trac)
        # calculate validation acc
        test_pred = model.predict(X_test)
        test_pred = [int(round(s)) for s in test_pred]
        teac = np.sum(y_test == test_pred)/y_test.shape[0]
        val_acc.append(teac)

    mean_train_acc = np.mean(train_acc)
    std_train_acc = np.std(train_acc)
    mean_val_acc = np.mean(val_acc)
    std_val_acc = np.std(val_acc)
    
    return mean_train_acc, std_train_acc, mean_val_acc, std_val_acc

def RF_GridSearch(params, X_train, y_train):
    X_col = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'b_image',
            'nmask', 'n3sig7', 'gauss_amp', 'gauss_R', 'abs_pv']
    # make sure the training set has the correct format
    X_train = X_train[X_col]

    best_acc = 0
    for k in params['max_features']:
        for j in params['min_samples_leaf']:
            print("Fitting for (max_features, min_samples_leaf)=({}, {})...".format(k,j))
            rf = RandomForestRegressor(n_estimators=150, max_features=k, 
                               min_samples_leaf=j, random_state=425, 
                               n_jobs=-1, verbose=0)
            train_acc, train_err, val_acc, val_err = X_validation(rf, X_train[X_col], y_train, cv=5)
            print("train acc:{}+/-{}, val acc:{}+/-{}".format(train_acc, train_err, val_acc, val_err))
            if val_acc > best_acc:
                best_acc = val_acc
                best_parameters = {'max_features':k, 
                                   'min_samples_leaf':j, 
                                   'train_acc':train_acc,
                                   'train_acc_err':train_err,
                                   'val_acc':val_acc,
                                   'val_acc_err':val_err
                                  }
                print("Updating best hyperparameter set:{}".format(best_parameters))

    rf = RandomForestRegressor(n_estimators=150, 
                               max_features=best_parameters['max_features'], 
                               min_samples_leaf=best_parameters['min_samples_leaf'], 
                               random_state=425, n_jobs=-1, verbose=0)

   
    rf.fit(X_train[X_col], y_train)
    
    pickle.dump(rf, open('rf.m', 'wb'))
    f = open("RF_best_param.txt","w")
    f.write( str(best_parameters) )
    f.close()

def CalcGTR(filename, model='RF'):
    df = fits2df(filename, 'DIFFERENCE_DETAB')
    col = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'b_image',
            'nmask', 'n3sig7', 'gauss_amp', 'gauss_R', 'abs_pv']
    X = df[col]

    if model == 'RF':
        m = pickle.load(open(getattr(config, 'rf_path'), 'rb'))
        gtr_score = m.predict(X)
        df['GTR_score'] = gtr_score
        FitsOp(filename, 'DIFFERENCE_DETAB', df, mode='update')
