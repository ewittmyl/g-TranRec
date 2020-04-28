import os
from . import config
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.wcs import WCS
from astropy.io.fits import getheader, getdata, update
import pandas as pd 
import numpy as np
from .image_process import fits2df, FitsOp, image_extract
from astropy.nddata import Cutout2D
from scipy.optimize import curve_fit
import pickle
from .gaussian import chunk_fit
from multiprocessing import Process, cpu_count, Manager
import math
from .postprocess import CalcWeight
from .xmatch import XmatchGLADE, mp_check, astroquery_xmatch
import pkg_resources
from .database import GladeDB



class SExtractor():
    """
    This Class.Obj is used to run SExtractor on a given FITS image. For example, running on the 'r0123456.fits[IMAGE]',
    you can run the command below:

    `f = SExtractor('r0123456.fits', 'IMAGE').run(thresh=2)`

    The result table would be append into the original FITS under the extension name '<image_ext>_DETAB'.

    Parameters
    ----------
    filename: str
        FITS filename you want to run on.
    image_ext: str, optional
        Extension name of the image. (default='IMAGE')
    """
    def __init__(self, filename, image_ext='IMAGE'):
        self.filename = filename
        self.extname = image_ext

    def create_config(self, **kwargs):
        """
        This function is to creat the configuration file and the gaussian kernel for running the SExtractor. The 
        input parameter, 'thresh', indicates the value of the config parameters 'DETECT_THRESH' and 'ANALYSIS_THRESH'.
        The following files would be created:
        1. .gtr.sex
        2. .gauss_2.5_5x5.conv

        Parameters
        ----------
        thresh : int, optional
            The 'DETECTION_THRESH' and the 'ANALYSIS_THRESH' values of running SExtractor. (default=2)
        """
        # create default config file
        os.system("{} -d > .gtr.sex".format(getattr(config, 'sex_cmd')))

        # create config arguments
        conf_args = {}
        conf_args['PARAMETERS_NAME'] = '.gtr.param'
        conf_args['CATALOG_TYPE'] = 'FITS_LDAC'
        conf_args['FILTER_NAME'] = '.gauss_2.5_5x5.conv'
        conf_args['DETECT_THRESH'] = str(int(kwargs['thresh']))
        conf_args['ANALYSIS_THRESH'] = str(int(kwargs['thresh']))
        conf_args['PIXEL_SCALE'] = '1.24198'
        conf_args['BACK_TYPE'] = 'AUTO'
        conf_args['DETECT_MINAREA'] = '5'
        conf_args['DEBLEND_NTHRESH'] = str(int(kwargs['deblend_nthresh']))
        conf_args['DEBLEND_MINCONT'] = str(int(kwargs['deblend_mincont']))
        conf_args['VERBOSE_TYPE'] = 'QUIET'
        conf_args['CATALOG_NAME'] = '_'.join([self.filename.split(".")[0], self.extname,'sex.fits'])

        # create gaussian filter
        f = open('.gauss_2.5_5x5.conv', 'w')
        print("""CONV NORM
    # 5x5 convolution mask of a gaussian PSF with FWHM = 2.5 pixels.
    0.034673 0.119131 0.179633 0.119131 0.034673
    0.119131 0.409323 0.617200 0.409323 0.119131
    0.179633 0.617200 0.930649 0.617200 0.179633
    0.119131 0.409323 0.617200 0.409323 0.119131
    0.034673 0.119131 0.179633 0.119131 0.034673""", file=f)
        f.close()

        # create param file
        params = ['X_IMAGE', 'Y_IMAGE', 'FLAGS', 'ERRCXYWIN_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO', 'B_IMAGE', 'FLAGS_WIN', 'ELLIPTICITY', 'FWHM_IMAGE']
        f = open('.gtr.param', 'w')
        print('\n'.join(params), file=f)
        f.close()

        self.conf_args = conf_args

    def create_cmd(self):
        """
        This function is to create the shell command for running SExtractor by joining the strings including
        image filename, config filename, etc. The command string would be stored in 'self.command'.
        """
        fn = '_'.join([self.filename.split('.')[0], self.extname + '.fits'])
        self.ext_filename = fn
        cmd = ' '.join([getattr(config, 'sex_cmd'), fn+'[0]', '-c .gtr.sex '])
        args = [''.join(['-', key, ' ', str(self.conf_args[key])]) for key in self.conf_args]
        cmd += ' '.join(args)

        self.command = cmd
    
    def run(self, thresh=2, deblend_nthresh=32, deblend_mincont=0.005):
        """
        This is the core function to run the SExtractor by executing the command generated by 'self.create_cmd()'. The 
        photometry table geneated by the SExtractor includes the following measurements
        (not ordering as below order):
        1. ra
        2. dec
        3. X_IMAGE
        4. Y_IMAGE
        5. FLAGS
        6. ERRCXYWIN_IMAGE
        7. MAG_AUTO
        8. MAGERR_AUTO
        9. mag
        10. B_IMAGE
        11. FLAGS_WIN
        12. ELLIPTICITYi
        13. FWHM_IMAGE

        The following files would be generated:
        1. <filename_basename>_sex.fits

        Parameters
        ----------
        thresh : int, optional
            The 'DETECTION_THRESH' and the 'ANALYSIS_THRESH' values of running SExtractor. (default=2)
        """

        image_extract(self.filename, extname=self.extname)

        # create config files 
        self.create_config(thresh=thresh, deblend_nthresh=deblend_nthresh, deblend_mincont=deblend_mincont)
        
        # create shell command
        self.create_cmd()

        # run SExtractor
        print("Running SExtractor on {}[{}]...".format(self.filename, self.extname))
        os.system(self.command)

        # remove all config files
        conf_files = ['.gtr.sex', '.gtr.param', '.gauss_2.5_5x5.conv', self.ext_filename]
        for f in conf_files:
            os.remove(f)
        
        # load DETECTION_TABLE
        intermediate_fn = self.conf_args['CATALOG_NAME']
        detection_table = fits2df(intermediate_fn, "LDAC_OBJECTS")

        # add (ra, dec) to DETECTION_TABLE
        w = WCS(self.filename)
        wcs = []
        for c in detection_table.iterrows():
            wcs.append(w.all_pix2world(c[1]['X_IMAGE']-1, c[1]['Y_IMAGE']-1, 0))
        wcs = pd.DataFrame(wcs, columns=['ra','dec'])
        final_det_tab = wcs.join(detection_table).astype('float')
        final_det_tab['x'] = final_det_tab['X_IMAGE']
        final_det_tab['y'] = final_det_tab['Y_IMAGE']

        # photometric calibration
        hdr = getheader(self.filename, 'IMAGE')
        final_det_tab['mag'] = hdr['CALAP']*final_det_tab['MAG_AUTO']+hdr['CALZP']

        # add extension table to input FITS
        if self.extname == 'IMAGE':
            FitsOp(self.filename, 'PHOTOMETRY', final_det_tab, mode='append')
        elif self.extname == 'DIFFERENCE':
            FitsOp(self.filename, 'PHOTOMETRY_DIFF', final_det_tab, mode='append')
        
        # for testing the optimal thresh and minarea
        # FitsOp(self.filename.split(".")[0]+'_sex.fits', '_'.join([self.extname, 'DETAB']), final_det_tab, mode='write')

        # remove intermediate file
        cmd = ' '.join(['rm', '-rf', intermediate_fn])
        os.system(cmd)


class CalcALL():
    def __init__(self, filename):
        self.filename = filename
        self.parameters = {
            'sciphoto': 'PHOTOMETRY',
            'sciimage': 'IMAGE',
            'diffphoto': 'PHOTOMETRY_DIFF',
            'diffimage': 'DIFFERENCE',
            'datapath': pkg_resources.resource_filename('gTranRec', 'data'),
        }

    def stamping(self, image_type='difference'):
        # get the coordinates for all detections on the difference image
        if image_type=='difference':
            self.stamps =  np.array([Cutout2D(self.diffimg, (self.diffphoto.iloc[i]['x']-1, self.diffphoto.iloc[i]['y']-1), 
                            (21, 21), mode='partial').data.reshape((21, 21)) for i in np.arange(self.diffphoto.shape[0])])
        elif image_type=='science':
            self.stamps =  np.array([Cutout2D(self.sciimg, (self.sciphoto.iloc[i]['x']-1, self.sciphoto.iloc[i]['y']-1), 
                            (21, 21), mode='partial').data.reshape((21, 21)) for i in np.arange(self.sciphoto.shape[0])])

    def cleaning(self):
        print("Data cleaning...")
        self.stamps = self.stamps.reshape(-1, 441)
        # replace all 0 by NaN
        self.stamps[self.stamps == 0] = np.nan
        # calculate the median noise level for each row (detection)
        row_median = np.nanmedian(self.stamps, axis=1)
        # find indicies that you need to replace
        inds = np.where(np.isnan(self.stamps))
        # replace all NaN by the detection median
        self.stamps[inds] = np.take(row_median, inds[0])

    def normalize_stamps(self):
        print("Normalizing thumbnails...")
        # flatten the stamp array
        flat_stamps = self.stamps.reshape(-1, 441)
        # calculate p-med(p)
        diff = flat_stamps-np.repeat(np.median(flat_stamps, axis=1)+1e-6, 441).reshape((flat_stamps.shape[0], 441))
        # calculate |p-med(p)|/sigma
        s2n = np.abs(diff)/np.repeat(np.std(flat_stamps, axis=1), 441).reshape((flat_stamps.shape[0], 441))
        self.norm_stamps = np.sign(diff)*np.log10(1+s2n)
        self.norm_stamps = np.nan_to_num(self.norm_stamps)

        self.norm_stamps = self.norm_stamps.reshape(-1, 21, 21)

    def calc_n3sig7(self):
        # count number of pixels with value less than -3sigma level over the 7x7 stamps
        sigma3 = np.std(self.norm_stamps.reshape(-1,441), axis=1)
        sigma3_matrix = np.repeat(-3*sigma3, 49).reshape((sigma3.shape[0], 49))
        n3sig7 = np.nansum(self.norm_stamps[:,21//2-3:21//2+4, 21//2-3:21//2+4].reshape(-1, 49) < sigma3_matrix, axis=1)
        return n3sig7

    def calc_gauss(self):
        # fit gauss
        print("Fitting 2D Gaussian...")
        # calculate the number of jobs per CPU
        num_jobs = math.ceil(self.norm_stamps.shape[0] / cpu_count())
        nstamps_chunks = [self.norm_stamps[x:x+num_jobs] for x in range(0, self.norm_stamps.shape[0], num_jobs)]

        # define return values from each processor
        gauss_amp = Manager()
        gauss_r = Manager()
        amp_dict = gauss_amp.dict()
        r_dict = gauss_r.dict()
        jobs = []
        for i in range(cpu_count()):
            print(i, nstamps_chunks[i].shape)
            p = Process(target=chunk_fit, args=(i,nstamps_chunks[i],amp_dict,r_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        g_amp, g_r = [], []
        for i in range(cpu_count()):
            g_amp += amp_dict[i]
            g_r += r_dict[i]

        return g_amp, g_r

    def PCA_transform(self):
        pca_pkl = '/'.join([self.parameters['datapath'], 'pca.m'])
        self.pca = pickle.load(open(pca_pkl, 'rb'))

        pca_col = ['pca{}'.format(i) for i in range(1, self.pca.components_.shape[0]+1)]
        df = pd.DataFrame(self.norm_stamps.reshape(-1, 441))
        pca_X = df.copy()
        pca_X = self.pca.transform(pca_X)
        pca_X = pd.DataFrame(pca_X, columns=pca_col)

        return pca_X

    def scidiff_offset(self):
        input_param = {
            'n_sig': 2, 
            'ang_sol': 1.24,
        }
        real_df = self.diffphoto[self.diffphoto.gtr_score > self.thresh]
        bogus_df = self.diffphoto[self.diffphoto.gtr_score < self.thresh]

        diff_det_coor = SkyCoord(ra=(real_df['ra']*u.degree).values, dec=(real_df['dec']*u.degree).values)
        sci_det_coor = SkyCoord(ra=(self.sciphoto['ra']*u.degree).values, dec=(self.sciphoto['dec']*u.degree).values)
        _, d2d, _ = diff_det_coor.match_to_catalog_sky(sci_det_coor)
        d2d = Angle(d2d, u.arcsec).arcsec
        real_df['scidiff_offset'] = d2d
        bogus_df['scidiff_offset'] = np.nan

        sig = np.mean(self.sciphoto['FWHM_IMAGE']) * input_param['ang_sol']
        real_df['scidiff_w'] = np.exp(-(d2d/(input_param['n_sig'] * sig)))
        bogus_df['scidiff_w'] = 0

        self.diffphoto = real_df.append(bogus_df, ignore_index = True)

    def calc_weight(self):
        input_param = {
            'n_sig': 3, 
            'ang_sol': 1.24,
        }
        sig = np.mean(self.sciphoto['FWHM_IMAGE']) * input_param['ang_sol']

        scidiff_w = self.diffphoto.scidiff_w
        gal_offset = self.diffphoto.GLADE_offset
        gal_offset = np.nan_to_num(gal_offset, nan=100)

        weight = (1-scidiff_w)*np.exp(-gal_offset/(input_param['n_sig']*sig))+scidiff_w

        self.diffphoto['weight'] = weight

    
    def run(self, thresh=0.5):
        
        self.thresh = thresh

        # load all useful tables
        self.diffphoto = fits2df(self.filename, self.parameters['diffphoto'])
        self.sciphoto = fits2df(self.filename, self.parameters['sciphoto'])
        self.glade = GladeDB.image_search(self.filename)
        self.fwhm = self.sciphoto.FWHM_IMAGE.median()

        # load difference images
        self.diffimg = getdata(self.filename, self.parameters['diffimage'])
        
        # make stamp for each detections on the difference image
        self.stamping()

        # define feature table
        self.X = pd.DataFrame()
        self.X['b_image'] = self.diffphoto['B_IMAGE']

        # create features
        self.X['nmask'] = np.nansum(self.stamps[:,21//2-3:21//2+4, 21//2-3:21//2+4].reshape(-1,49)==0, axis=1)
        self.X['nmask'] = self.X['nmask'] // 10

        # data cleaning
        self.cleaning()

        # normalize the stamps
        self.normalize_stamps()

        # create features
        self.X['n3sig7'] = self.calc_n3sig7()
        self.X['abs_pv'] = np.nansum(np.abs(self.norm_stamps.reshape(-1, 441)), axis=1)

        # fitting gaussian to stamps
        g_amp, g_r = self.calc_gauss()

        # best fit gaussian amplitude of the detection
        self.X['gauss_amp'] = g_amp

        # R-squared statistic of the best fit gaussian
        self.X['gauss_R'] = g_r

        self.X = self.X.join(self.PCA_transform())

        col = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'b_image',
                'nmask', 'n3sig7', 'gauss_amp', 'gauss_R', 'abs_pv']

        self.X = self.X[col]

        classifier_pkl = '/'.join([self.parameters['datapath'], 'rf.m'])
        self.classifier = pickle.load(open(classifier_pkl, 'rb'))
        print("Calculating GTR score...")

        gtr_score = self.classifier.predict(self.X)
        self.diffphoto['gtr_score'] = gtr_score

        self.diffphoto = XmatchGLADE(self.diffphoto, self.glade.copy(), self.thresh)
            

        self.scidiff_offset()

        self.calc_weight()
        self.diffphoto['gtr_wscore'] = self.diffphoto.weight * self.diffphoto.gtr_score

        # self.diffphoto = mp_check(self.filename, self.diffphoto, self.thresh)

        # try:
        #     radius = ( self.fwhm * 1.24 ) / 2
        #     xmatch_df = astroquery_xmatch(self.diffphoto, r=radius, GTR_thresh=self.thresh)
        #     self.diffphoto = xmatch_df
        # except:
        #     print("Cannot X-match with NED and SIMBAD catalog...")
            
        # self.diffphoto.drop(columns=self.diffphoto.columns[self.diffphoto.dtypes=='object'], inplace=True)


        FitsOp(self.filename, 'PHOTOMETRY_DIFF', self.diffphoto, mode='update')
    
    def save_features(self, image_type):
        
        if image_type == 'difference':
            # load all useful tables
            self.diffphoto = fits2df(self.filename, self.parameters['diffphoto'])
            self.diffimg = getdata(self.filename, self.parameters['diffimage'])
        elif image_type == 'science':
            # load difference images
            self.sciphoto = fits2df(self.filename, self.parameters['sciphoto'])
            self.sciimg = getdata(self.filename, self.parameters['sciimage'])
            
        # make stamp for each detections on the difference image
        self.stamping(image_type=image_type)

        # define feature table
        self.X = pd.DataFrame()
        if image_type == 'science':
            self.X['b_image'] = self.sciphoto['B_IMAGE']
        elif image_type == 'difference':
            self.X['b_image'] = self.diffphoto['B_IMAGE']

        # create features
        self.X['nmask'] = np.nansum(self.stamps[:,21//2-3:21//2+4, 21//2-3:21//2+4].reshape(-1,49)==0, axis=1)
        self.X['nmask'] = self.X['nmask'] // 10

        # data cleaning
        self.cleaning()

        # normalize the stamps
        self.normalize_stamps()

        # create features
        self.X['n3sig7'] = self.calc_n3sig7()
        self.X['abs_pv'] = np.nansum(np.abs(self.norm_stamps.reshape(-1, 441)), axis=1)

        # fitting gaussian to stamps
        g_amp, g_r = self.calc_gauss()

        # best fit gaussian amplitude of the detection
        self.X['gauss_amp'] = g_amp

        # R-squared statistic of the best fit gaussian
        self.X['gauss_R'] = g_r

        if image_type=='science':
            new_photo = self.sciphoto.join(self.X)
        elif image_type=='difference':
            new_photo = self.diffphoto.join(self.X)
        
        new_photo.drop(columns=new_photo.columns[new_photo.dtypes=='object'], inplace=True)

        if image_type=='science':
            FitsOp(self.filename, 'PHOTOMETRY', new_photo, mode='update')
        elif image_type=='difference':
            FitsOp(self.filename, 'PHOTOMETRY_DIFF', new_photo, mode='update')

    def save_pca(self, image_type):

        if image_type == 'difference':
            # load all useful tables
            self.diffphoto = fits2df(self.filename, self.parameters['diffphoto'])
            self.diffimg = getdata(self.filename, self.parameters['diffimage'])
        elif image_type == 'science':
            # load difference images
            self.sciphoto = fits2df(self.filename, self.parameters['sciphoto'])
            self.sciimg = getdata(self.filename, self.parameters['sciimage'])
            
        # make stamp for each detections on the difference image
        self.stamping(image_type=image_type)

        # data cleaning
        self.cleaning()

        # normalize the stamps
        self.normalize_stamps()

        if image_type=='science':
            new_photo = self.sciphoto.join(self.PCA_transform())
        elif image_type=='difference':
            new_photo = self.diffphoto.join(self.PCA_transform())

        new_photo.drop(columns=new_photo.columns[new_photo.dtypes=='object'], inplace=True)

        if image_type=='science':
            FitsOp(self.filename, 'PHOTOMETRY', new_photo, mode='update')
        elif image_type=='difference':
            FitsOp(self.filename, 'PHOTOMETRY_DIFF', new_photo, mode='update')

