import os
from . import config
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.io.fits import getheader, getdata, update
import pandas as pd 
import numpy as np
from .image_process import image_extract
from astropy.nddata import Cutout2D
from scipy.optimize import curve_fit
import pickle
from .gaussian import chunk_fit
from multiprocessing import Process, cpu_count, Manager
import math


def FitsOp(filename, extname, dataframe, mode='append'): 
    """
    This function is to perform basic FITS operation, which includes appending, writing and updating extension table on the 
    given FITS file.

    Parameters
    ----------
    filename: str
        FITS filename you want to operate on.
    extname: str
        Extension name of the table you want to operate on.
    dataframe: pd.DataFrame
        Pandas dataframe used to operate.
    mode: str, optional
        ['append'/'write'/'update']
        Operation type. (default='append')
    """
    if mode == 'append':  
        # add extension table to input FITS
        print("Adding new extension table {} into {}...".format(extname, filename))
        m = Table(dataframe.values, names=dataframe.columns)
        hdu = fits.table_to_hdu(m)
        with fits.open(filename, mode='update') as hdul0:
            hdul0.append(hdu)
            hdul0[-1].header['EXTNAME'] = extname
            hdul0.flush()
    elif mode == 'write':
        print("Creating new FITS {}...".format(filename))
        astropy_tab = Table.from_pandas(dataframe)
        astropy_tab.write(filename, format='fits')
        with fits.open(filename, mode='update') as hdul0:
            hdul0[-1].header['EXTNAME'] = extname
            hdul0.flush()
    elif mode == 'update':
        print("Updating {}[{}]...".format(filename, extname))
        m = Table(dataframe.values, names=dataframe.columns)
        hdr = getheader(filename, extname=extname)
        update(filename, np.array(m), extname=extname, header=hdr)


def fits2df(filename, extname):
    """
    This function is to load the FITS extension table and convert to Pandas DataFrame.

    Parameters
    ----------
    filename: str
        FITS filename you want to load.
    extname: str
        Extension name of the table you want to load.

    Return
    ------
    out: pd.DataFrame
        Extension table of the FITS file with DataFrame format.
    """
    df = getdata(filename, extname)
    df = pd.DataFrame(np.array(df).byteswap().newbyteorder())
    return df

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

        # photometric calibration
        hdr = getheader(self.filename, 'IMAGE')
        final_det_tab['mag'] = hdr['CALAP']*final_det_tab['MAG_AUTO']+hdr['CALZP']

        # add extension table to input FITS
        FitsOp(self.filename, '_'.join([self.extname, 'DETAB']), final_det_tab, mode='append')
        
        # for testing the optimal thresh and minarea
        # FitsOp(self.filename.split(".")[0]+'_sex.fits', '_'.join([self.extname, 'DETAB']), final_det_tab, mode='write')

        # remove intermediate file
        cmd = ' '.join(['rm', '-rf', intermediate_fn])
        os.system(cmd)

class FeatureExtract():
    """
    This Class.Obj is used to create the feature table for performing machine learning. It takes the extension table 
    '<image_ext>_DETAB' in the FITS as an input. Therefore, make sure you have already run 'SExtractor().run()' to create 
    the detection table for the image first. 

    Parameters
    ----------
    filename: str
        FITS filename you want to run on.
    image_ext: str, optional
        Extension name of the image. (default='DIFFERENCE')
    """
    def __init__(self, filename, image_ext='DIFFERENCE'):
        # rename the filename 
        self.filename = filename
        # indicate image_type
        self.image_type = image_ext
        # recall detection table extname
        self.detab_extname = '_'.join([image_ext, 'DETAB'])
        # read-in detection table
        self.detab = fits2df(filename, self.detab_extname)
        # load image
        self.image = getdata(filename, image_ext)
        # create stamps
        self.stamps = np.array([Cutout2D(self.image, (self.detab.iloc[i]['X_IMAGE']-1, self.detab.iloc[i]['Y_IMAGE']-1), 
                    (21, 21), mode='partial').data.reshape((21, 21)) for i in np.arange(self.detab.shape[0])])

    def data_cleaning(self):
        print("Data cleaning...")
        self.stamps = self.stamps.reshape(-1, 441)
        # replace all 0 by NaN
        self.stamps[self.stamps == 0] = np.nan
        # calculate the median noise level for each row (detection)
        row_median = np.nanmedian(self.stamps, axis=1)
        #Find indicies that you need to replace
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
        self.norm_stamps = self.norm_stamps.reshape(-1, 21, 21)

    
    def make_X(self):
        print("Creating feature table for {}[{}]...".format(self.filename, self.image_type))
        # initialize feature table X
        self.X = pd.DataFrame()
        # semiminor axis of the detection
        self.X['b_image'] = self.detab.B_IMAGE
        # number of masked pixel over the 7x7 box
        self.X['nmask'] = np.nansum(self.stamps[:,21//2-3:21//2+4, 21//2-3:21//2+4].reshape(-1,49)==0, axis=1)
        self.X['nmask'] = self.X['nmask'] // 10
        
        # data cleaning
        self.data_cleaning()
        # normalize the stamp
        self.normalize_stamps()
        # number of pixels with value less than -3sigma level over the 7x7 box
        sigma3 = np.std(self.norm_stamps.reshape(-1, 441), axis=1)
        sigma3_matrix = np.repeat(-3*sigma3, 49).reshape((sigma3.shape[0], 49))
        self.X['n3sig7'] = np.nansum(self.norm_stamps[:,21//2-3:21//2+4, 21//2-3:21//2+4].reshape(-1, 49) < sigma3_matrix, axis=1)

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
            p = Process(target=chunk_fit, args=(i,nstamps_chunks[i],amp_dict,r_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        g_amp, g_r = [], []
        for i in range(cpu_count()):
            g_amp += amp_dict[i]
            g_r += r_dict[i]
        
        # best fit gaussian amplitude of the detection
        self.X['gauss_amp'] = g_amp
        # R-squared statistic of the best fit gaussian
        self.X['gauss_R'] = g_r
        # sum of the absolute pixel values over the entire stamp
        self.X['abs_pv'] = np.nansum(np.abs(self.norm_stamps.reshape(-1,441)), axis=1)
        # join X into detection table
        self.detab = self.detab.join(self.X)
        self.X_col = ['b_image','nmask','n3sig7','gauss_amp','gauss_R','abs_pv']
        FitsOp(self.filename, '_'.join([self.image_type, 'DETAB']), self.detab, mode='update')

    def make_PCA(self):
        if not 'norm_stamps' in dir(self):
            # data cleaning
            self.data_cleaning()
            # normalize the stamp
            self.normalize_stamps()
        # load trained PCA and KBest
        pca = pickle.load(open(getattr(config, 'pca_path'), 'rb'))
        pca_col = ['pca{}'.format(i) for i in range(1,pca.components_.shape[0]+1)]
        try:
            self.detab.drop(columns=pca_col, inplace=True)
        except:
            pass

        df = pd.DataFrame(self.norm_stamps.reshape(-1,441))
        pca_X = df.copy()
        # PCA tranformation to stamps
        pca_X = pca.transform(pca_X)
        print("Adding best PCA features into {}[{}_DETAB]".format(self.filename, self.image_type))
        self.pca_X = pd.DataFrame(pca_X, columns=pca_col)

        self.detab = self.detab.join(self.pca_X)

        FitsOp(self.filename, '_'.join([self.image_type, 'DETAB']), self.detab, mode='update')


    def store_stamps(self):
        df = pd.DataFrame(self.norm_stamps.reshape(-1,441))
        FitsOp(self.filename, '_'.join([self.image_type, 'STAMPS']), df, mode='append')




    