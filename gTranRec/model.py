from . import config
from .image_process import scaling, uncompress
import pkg_resources
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from keras.models import load_model
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from keras import models, layers, optimizers, regularizers
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from astropy.io.fits import getdata
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.io.fits import update, getheader
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from keras.layers import Dense, Conv2D, Flatten 
from keras.layers import Dropout, Activation, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from astropy.visualization import ZScaleInterval
from sklearn.feature_selection import SelectKBest
from .sex import get_stamp
from sklearn.decomposition import PCA

algorithm_dict = {'rf': 'Random_Forest',
                'pca_rf': 'PCA_Random_Forest',
                'ann': 'Artificial_Neural_Network',
                'cnn': 'Convolutional_Neural_Network'}

class classifier():
        def __init__(self, model, algorithm, pca, kbest):
                self.model = model
                self.algorithm = algorithm_dict[algorithm]
                self.pca = pca
                self.kbest = kbest

        @classmethod
        def _load_model(cls, model_path=None, algorithm='rf'):
                if not model_path:
                        DATA_DIR = getattr(config, 'DATA_DIR')
                        filename = '.'.join([algorithm, 'm'])
                        MODEL_PATH = os.path.join(DATA_DIR, filename)
                        if (algorithm == 'rf') or (algorithm == 'pca_rf'):
                                model = pickle.load(open(MODEL_PATH, 'rb'))
                        elif (algorithm == 'ann') or (algorithm == 'cnn'):
                                model = load_model(MODEL_PATH)
                else:
                        if (algorithm == 'rf') or (algorithm == 'pca_rf'):
                                model = pickle.load(open(model_path, 'rb'))
                        elif (algorithm == 'ann') or (algorithm == 'cnn'):
                                model = load_model(model_path)

                if algorithm == 'pca_rf':
                        if model_path:
                                data_dir = os.path.dirname(model_path)
                                
                                pca_path = os.path.join(data_dir, 'pca.p')
                                pca = pickle.load(open(pca_path+'', 'rb'))

                                kbest_path = os.path.join(data_dir, 'kbest.tran')
                                kbest = pickle.load(open(kbest_path, 'rb'))
                        else:
                                DATA_DIR = getattr(config, 'DATA_DIR')

                                pca_path = os.path.join(DATA_DIR, 'pca.p')
                                pca = pickle.load(open(pca_path+'', 'rb'))

                                kbest_path = os.path.join(DATA_DIR, 'kbest.tran')
                                kbest = pickle.load(open(kbest_path, 'rb'))
                else:
                        pca = None
                        kbest = None

                return cls(model, algorithm, pca, kbest)

        def predict_all(self, filename, extname={'image':'IMAGE', 'table':'DETECTION_TABLE'}):
                det_tab = pd.DataFrame(np.array(getdata(filename, extname['table'])).byteswap().newbyteorder())

                # clean data with inf or NaN
                det_tab = det_tab.replace([np.inf, -np.inf], np.nan)
                det_tab.dropna(inplace=True)

                # load image
                pix_val = getdata(filename, extname['image'])

                # create stamp DataFrame
                pixel_col = ['p'+str(i+1) for i in np.arange(441)]
                stamps = [Cutout2D(pix_val, (det_tab.iloc[i]['X_IMAGE']-1, det_tab.iloc[i]['Y_IMAGE']-1), 
                        (21, 21), mode='partial').data.reshape(441) for i in np.arange(det_tab.shape[0])]
                stamps = pd.DataFrame(stamps, columns=pixel_col)

                # scale detection stamps
                X = scaling(stamps)

                if self.algorithm == 'PCA_Random_Forest':
                        X = self.pca.transform(X)
                        X = self.kbest.transform(X)

                # make predictions
                if (self.algorithm == 'Random_Forest') or (self.algorithm == 'PCA_Random_Forest'):
                        pred = self.model.predict(X)
                elif (self.algorithm == 'Artificial_Neural_Network') or (self.algorithm == 'Convolutional_Neural_Network'):
                        pred = self.model.predict(X)[:,1]
                        
                det_tab['real_bogus'] = pred

                # update the 'DETECTION_TABLE' including real-bogus score
                m = Table(det_tab.values, names=det_tab.columns)
                hdr = getheader(filename, extname='DETECTION_TABLE')
                update(filename, np.array(m), extname='DETECTION_TABLE', header=hdr)

        def predict_single(self, filename, coord, image_ext='IMAGE', coord_sys='image', inspect=True):

                # load image
                pix_val = getdata(filename, image_ext)

                global x, y
                if coord_sys == 'image':
                        x = coord[0] + 1
                        y = coord[1] + 1
                elif coord_sys == 'wcs':
                        # convert world coordinates to image coordinates if coord_sys=='wcs'
                        w = WCS(filename)
                        x, y = w.all_world2pix(coord[0], coord[1], 0)
                        # convert astropy coordinate (start from 0,0) to SExtractor coordiate (start from 1,1)
                        x += 1
                        y += 1
                else:
                        raise KeyError("Argument 'coord_sys' can only be 'wcs' or 'image'!")

                # create stamps DataFrame
                pixel_col = ['p'+str(i+1) for i in np.arange(441)]
                stamp = [Cutout2D(pix_val, (x-1, y-1), 
                                (21, 21), mode='partial').data.reshape(441)]
                stamp = pd.DataFrame(stamp, columns=pixel_col)

                # scale stamp
                X = scaling(stamp)

                if self.algorithm == 'PCA_Random_Forest':
                        X = self.pca.transform(X)

                # make predictions
                if (self.algorithm == 'Random_Forest') or (self.algorithm == 'PCA_Random_Forest'):
                        score = self.model.predict(X)[0]
                elif (self.algorithm == 'Artificial_Neural_Network') or (self.algorithm == 'Convolutional_Neural_Network'):
                        score = self.model.predict(X)[0][1]
                        

                # plot the window image if inspect=True
                if inspect:
                        # plot the window image and print out the score
                        print("Score: {}".format(score))
                        interval = ZScaleInterval()
                        stamp = [Cutout2D(pix_val, (x-1, y-1), (150, 150), mode='partial').data.reshape(22500)]
                        stamp = interval(stamp)
                        img = stamp[0].reshape(150,150)
                        fig, ax = plt.subplots(1)
                        ax.imshow(img, cmap="gray")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.axhline(75, xmin=0.4, xmax=0.45,color='red', linewidth=2)
                        ax.axhline(75, xmin=0.55, xmax=0.6,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.4, ymax=0.45,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.55, ymax=0.6,color='red', linewidth=2)
                        plt.show()

                # return the real score
                return score


class data_set():
        """
        For building up the data set for training or testing. The DataFrame contains:
        1. 441 pixel values (p1-p441)
        2. image coordinates (x, y)
        3. filename of the sample extracted (filename)
        4. label of real or bogus (target)
        """
        def __init__(self, dataframe):
                self.dataframe = dataframe

        @classmethod
        def create_new(cls):
                """
                Creating empty pandas DataFrame with defined columns.
                """
                col = ['p'+str(i+1) for i in np.arange(441)] + ['x', 'y', 'filename', 'target']
                df = pd.DataFrame(columns=col)
                return cls(df)

        @classmethod

        def _load_(cls, input="dataset.csv"):
                """
                Loading the CSV to create pandas DataFrame. The CSV table should contains columns of:
                1. p1-p441 (float)
                2. x, y (float)
                3. filename (str)
                4. target (int)
 
                Parameters:
                ----------
                input: str
                        input filename with CSV format
                """
                dataframe = pd.read_csv(input)
                return cls(dataframe)
                
        def _append_(self, filenames, label, thresh='1.5'):
                """
                To append the existing table with new data. SExtractor will be ran on the images.

                Parameters:
                ----------
                filenames: list of str
                        filenames of the image FITS
                label: str (real/bogus)
                        which sample the detections will be put into
                thresh: str of int
                        DETECT_THRESH of SExtractor
                """
                for fn in filenames:
                        uncompress(fn)
                        stamp_fn = get_stamp(fn, label, thresh=thresh)
                        feature_tab = pd.read_csv(stamp_fn)
                        self.dataframe = self.dataframe.append(feature_tab)
        
        def _save_(self, output="dataset.csv"):
                """
                Saving the DataFrame.

                Parameters:
                ----------
                output: str
                        output filename in CSV format
                """
                self.dataframe.to_csv(output, index=False)

class model_build():
        def __init__(self, model, algorithm):
                self.model = model
                self.algorithm = algorithm_dict[algorithm]

        @classmethod
        def create_new(cls, algorithm, par):
                """
                Create new untrained model.

                Parameters:
                ----------
                algorithm: str (rf/pca_rf/ann/cnn)
                        which algorithm will be used
                par: dict
                        For RF:
                        - n_estimators: suggest using 1000
                        - max_features: suggest using 25
                        For ANN:
                        - neurons: suggest using [200]
                        - l2_lambda: suggest using 0
                        For CNN:
                        - conv_nodes: suggest using [32, 128]
                        - dense_nodes: suggest using [512, 512]
                        - dropout_p: suggest using [0.1, 0.5]
                """
                seed = 369
                if (algorithm == 'rf') or (algorithm == 'pca_rf'):
                        model = RandomForestRegressor(n_estimators=par['n_estimators'], 
                                                max_features=par['max_features'], min_samples_leaf=1, 
                                                random_state=seed, n_jobs=-1)
                elif algorithm == 'ann':
                        model = models.Sequential()
                        model.add(layers.Dense(par['neurons'][0], activation='relu', input_shape=(441,), 
                                        kernel_regularizer=regularizers.l2(par['l2_lambda'])))
                        if len(par['neurons']) > 1:
                                # add more hidden layers if `len(neurons) > 1`
                                for n in par['neurons'][1:]:
                                        model.add(layers.Dense(n, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))

                        # add output layer 
                        model.add(layers.Dense(2, activation='softmax'))

                        # define optimizer = RMS
                        # low learning rate avoids over shoot of correction
                        optimizer = optimizers.RMSprop(lr=1e-4)

                        # compile model, using accuracy to fit training data
                        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                elif algorithm == 'cnn':
                        model = models.Sequential()

                        model.add(Conv2D(par['conv_nodes'][0], 3, 3, 
                                        activation='relu', input_shape=(21,21,1), dim_ordering="tf"))
                        model.add(MaxPooling2D((2, 2), dim_ordering="tf"))
                        if len(par['dropout_p']) != 0:
                                model.add(Dropout(par['dropout_p'][0]))
                        
                        if len(par['conv_nodes']) > 1:
                                for i in par['conv_nodes'][1:]:
                                        model.add(Conv2D(i, 3, 3, activation='relu', dim_ordering="tf"))
                                        model.add(MaxPooling2D((2, 2), dim_ordering="tf"))
                                        if len(par['dropout_p']) != 0:
                                                model.add(Dropout(par['dropout_p'][0]))
                        model.add(Flatten())

                        for n in par['dense_nodes']:
                                model.add(Dense(n))
                                model.add(Activation("relu"))
                                if len(par['dropout_p']) != 0:
                                        model.add(Dropout(par['dropout_p'][1]))
                                        
                        model.add(Dense(2))
                        model.add(Activation('softmax'))

                        lr = 0.1
                        decay = lr/50
                        optimizer = optimizers.SGD(lr=lr, decay=decay, nesterov=True)

                        model.compile(optimizer=optimizer, 
                                loss='categorical_crossentropy', 
                                metrics=['accuracy'])

                return cls(model, algorithm)
        
        def train_model(self, train_set, par, imbalancing=False):
                """
                Training model on the given training data set.

                train_set: pd.DataFrame
                        training data set
                par: dict
                        For PCA_RF:
                        - n_pc: suggest using 441
                        - kbest: suggest using 13
                        For ANN:
                        - epochs: suggest using 60
                        - batches: suggest using 2000
                        For CNN:
                        - epochs: suggest using 15
                        - batches: suggest using 5000
                imbalancing: bool
                        create 4:1 imbalance training set if True
                """
                if imbalancing:
                        # imbalacing the sample size to 4:1
                        real = train_set[train_set.target==1].sample(n=100000)
                        bogus = train_set[train_set.target==0]
                        train_set = real.append(bogus)

                X_train = scaling(train_set)
                y_train = train_set.target


                if 'PCA' in self.algorithm:
                        # pca transformation
                        pca = PCA(n_components=par['n_pc'])
                        pca.fit(X_train)
                        X_train = pca.transform(X_train)
                        self.pca = pca
                        # univariate statics
                        kbest_transform = SelectKBest(k=par['kbest'])
                        kbest_transform.fit(X_train, y_train)
                        X_train = kbest_transform.transform(X_train)
                        self.kbest_transform = kbest_transform
                
                if 'Neural_Network' in self.algorithm:
                        # extract target labels
                        train_labels = to_categorical(y_train)
                        name = self.algorithm+'-{}'.format(int(time.time()))
                        tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
                        mc = ModelCheckpoint(name+'.m', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                        # model training 
                        self.model.fit(X_train, train_labels, epochs=par['epochs'], 
                                        batch_size=par['batches'], validation_split=0.2, callbacks=[tensorboard, mc])
                elif 'Random_Forest' in self.algorithm:
                        # model training 
                        self.model.fit(X_train, y_train)
        
        def test_model(self, test_set):
                X_test = scaling(test_set)

                if 'PCA' in self.algorithm:
                        X_test = self.pca.transform(X_test)
                        X_test = self.kbest_transform.transform(X_test)

                if (self.algorithm == 'Random_Forest') or (self.algorithm == 'PCA_Random_Forest'):
                        pred = self.model.predict(X_test)
                elif (self.algorithm == 'Artificial_Neural_Network') or (self.algorithm == 'Convolutional_Neural_Network'):
                        pred = self.model.predict(X_test)[:,1]

                test_set['pred'] = pred

                return test_set
                

        def _save_(self):
                if self.algorithm == 'Random_Forest':
                        pickle.dump(self.model, open('rf.m', 'wb'))
                elif self.algorithm == 'PCA_Random_Forest':
                        pickle.dump(self.model, open('pca_rf.m', 'wb'))
                elif self.algorithm == 'Artificial_Neural_Network':
                        self.model.save('ann.m')
                elif self.algorithm == 'Convolutional_Neural_Network':
                        self.model.save('cnn.m')

                if 'PCA' in self.algorithm:
                        pickle.dump(self.pca, open('pca.p', 'wb'))
                        pickle.dump(self.kbest_transform, open('kbest.tran', 'wb'))
