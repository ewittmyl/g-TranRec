import os
from . import config
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.wcs import WCS
from astropy.io.fits import getheader
import pandas as pd 
from .image_process import fits2df, FitsOp, image_extract
import math
import pkg_resources

class SExtractor():
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