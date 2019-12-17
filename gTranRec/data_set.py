from .features import SExtractor, FeatureExtract
from .download import pull
from .image_process import unzip
from . import config
import os
import pandas as pd
import numpy as np
from .features import fits2df
from sklearn.utils import shuffle
import pickle
from .postprocess import make_Mag2Lim


def PrepImages(image_list, label='both'):
    images = [line.rstrip('\n') for line in open(image_list)]
    user = getattr(config, 'user')
    password = getattr(config, 'password')

    for img in images:
        fn = img + "-median.fits"
        if not os.path.isfile(fn):
            pull(user, password, fn)
            unzip(fn)
        if label=='both':
            SExtractor(fn, image_ext='IMAGE').run()
            sci_f = FeatureExtract(fn, 'IMAGE')
            sci_f.make_X()
            sci_f.store_stamps()
            SExtractor(fn, image_ext='DIFFERENCE').run()
            diff_f = FeatureExtract(fn, 'DIFFERENCE')
            diff_f.make_X()
            diff_f.store_stamps()
        elif label=='real':
            SExtractor(fn, image_ext='IMAGE').run()
            sci_f = FeatureExtract(fn, 'IMAGE')
            sci_f.make_X()
            sci_f.store_stamps()
        elif label=='bogus':
            SExtractor(fn, image_ext='DIFFERENCE').run()
            diff_f = FeatureExtract(fn, 'DIFFERENCE')
            diff_f.make_X()
            diff_f.store_stamps()

def AddPCA():
    images = [img for img in os.listdir("./") if '.fits' in img]
    for img in images:
        try:
            sci_f = FeatureExtract(img, 'IMAGE')
            sci_f.make_PCA()
        except:
            pass
        
        try:
            diff_f = FeatureExtract(img, 'DIFFERENCE')
            diff_f.make_PCA()
        except:
            pass

def AddMag2Lim():
    images = [img for img in os.listdir("./") if '.fits' in img]
    for img in images:
        try:
            make_Mag2Lim(img, image_type='IMAGE')
        except:
            pass
        
        try:
            make_Mag2Lim(img, image_type='DIFFERENCE')
        except:
            pass
        

def create_df(outfile='all_data.csv'):
    images = [img for img in os.listdir(".") if ".fits" in img]
    # load trained PCA and KBest
    pca = pickle.load(open(getattr(config, 'pca_path'), 'rb'))
    X_col = ['pca{}'.format(i) for i in range(1,pca.components_.shape[0]+1)] + ['b_image','nmask','n3sig7','gauss_amp','gauss_R','abs_pv','FLAGS','FLAGS_WIN','ERRCXYWIN_IMAGE', 'X_IMAGE', 'Y_IMAGE', 'y','mag2lim']
    df = pd.DataFrame(X_col)
    for img in images:
        try:
            print("Appending training set for {}".format(img))
            real_df = fits2df(img, 'IMAGE_DETAB')
            real_df['y'] = 1
            df = df.append(real_df[X_col], ignore_index = True)
        except:
            pass

        try:
            bogus_df = fits2df(img, 'DIFFERENCE_DETAB')
            bogus_df['y'] = 0
            df = df.append(bogus_df[X_col], ignore_index = True)
        except:
            pass
    df = df[X_col]
    mask = (df.X_IMAGE > 21) & (df.X_IMAGE < 8155) & (df.Y_IMAGE > 21) & (df.Y_IMAGE < 6111) & (df.y==1) & (df.FLAGS==0) & (df.FLAGS_WIN==0) & (abs(df.ERRCXYWIN_IMAGE)<100) 
    df = df[df.y==0].append(df[mask], ignore_index = True)
    df.dropna(inplace=True)
    df.drop(columns=['X_IMAGE', 'Y_IMAGE', 'FLAGS','FLAGS_WIN','ERRCXYWIN_IMAGE'], inplace=True)
    df.to_csv(outfile, index=False)


def data_sampling(infile='all_data.csv', outfile='bal_set.csv', real2bogus=[360000,360000]):
    df = pd.read_csv(infile)
    sub_df = df[df.y==0].sample(n=real2bogus[0], random_state=1).append(df[df.y==1].sample(n=real2bogus[1], random_state=1), ignore_index=True)
    sub_df = shuffle(sub_df)
    sub_df.to_csv(outfile, index=False)
