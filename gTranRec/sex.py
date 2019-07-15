import os
from . import config
from astropy.io.fits import getdata
import pandas as pd 
import numpy as np
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.io.fits import getheader
from .image_process import image_extract


def create_config(thresh='1.5', detect_minarea='9', deblend_nthresh='32', deblend_mincount='0.005'):
    """
    Creating config files for SExtractor: 
    1. .gtr.sex
    2. .gtr.param
    3. .gauss_2.5_5x5.conv

    Parameters:
    ----------
    thresh: str(int)
        DETECT_THRESH and ANALYSIS_THRESH
    detect_minarea: str(int)
        DETECT_MINAREA
    deblend_nthresh: str(int)
        DEBLEND_NTHRESH
    deblend_mincount: str(int)
        DEBLEND_MINCONT

    Return:
    ----------
    conf_args: dict
        dictionary of config arguments
    """
    # create default config file
    os.system("{} -d > .gtr.sex".format(getattr(config, 'sex_cmd')))


    # create config arguments
    conf_args = {}
    conf_args['PARAMETERS_NAME'] = '.gtr.param'
    conf_args['CATALOG_TYPE'] = 'FITS_LDAC'
    conf_args['FILTER_NAME'] = '.gauss_2.5_5x5.conv'
    conf_args['DETECT_THRESH'] = thresh
    conf_args['ANALYSIS_THRESH'] = thresh
    conf_args['PIXEL_SCALE'] = '1.24198'
    conf_args['BACK_TYPE'] = 'AUTO'
    conf_args['DETECT_MINAREA'] = detect_minarea
    conf_args['DEBLEND_NTHRESH'] = deblend_nthresh
    conf_args['DEBLEND_MINCONT'] = deblend_mincount

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
    params = ['X_IMAGE', 'Y_IMAGE', 'FLAGS', 'ERRCXYWIN_IMAGE', 'MAG_AUTO', 'MAGERR_AUTO']
    f = open('.gtr.param', 'w')
    print('\n'.join(params), file=f)
    f.close()

    return conf_args

def get_cmd(filename, conf_args):
    """
    Creating SExtractor command for running.

    Parameters:
    ----------
    filename: str
        filename of the image FITS
    conf_args: dict
        object created by 'create.config'

    Return:
    ----------
    cmd: str
        SExtractor command
    """
    cmd = ' '.join([getattr(config, 'sex_cmd'), filename+'[0]', '-c .gtr.sex '])
    conf_args['CATALOG_NAME'] = '_'.join([filename.split(".")[0], 'cand.fits'])
    args = [''.join(['-', key, ' ', str(conf_args[key])]) for key in conf_args]
    cmd += ' '.join(args)
    return cmd

def config_cleanup():
    """
    Removing all config files after running SExtractor.
    """
    conf_files = ['.gtr.sex', '.gtr.param', '.gauss_2.5_5x5.conv']
    for f in conf_files:
        os.remove(f)

def run_sex(filename, thresh='1.5', detect_minarea='5',deblend_nthresh='32', deblend_mincount='0.005'):
    """
    Running SExtractor on the FITS image and creating the DETECTION_TABLE. 

    Attributes in the DETECTION_TABLE:
    1. ALPHA_J2000
    2. DELTA_J2000
    3. X_IMAGE
    4. Y_IMAGE
    5. FLAGS
    6. ERRCXYWIN_IMAGE
    7. MAG_AUTO
    8. MAGERR_AUTO
    9. mag
    
    Parameters:
    ----------
    filename: str
        filename of the image FITS
    thresh: str(int)
        DETECT_THRESH and ANALYSIS_THRESH
    detect_minarea: str(int)
        DETECT_MINAREA
    deblend_nthresh: str(int)
        DEBLEND_NTHRESH
    deblend_mincount: str(int)
        DEBLEND_MINCONT
    """
    # create config files
    conf_args = create_config(thresh=thresh, detect_minarea=detect_minarea, deblend_nthresh=deblend_nthresh, deblend_mincount=deblend_mincount)

    # run SExtractor
    cmd = get_cmd(filename, conf_args)
    os.system(cmd)

    # remove all config files
    config_cleanup()

    # load DETECTION_TABLE
    intermediate_fn = '_'.join([filename.split(".")[0], 'cand.fits'])
    detection_table = getdata(intermediate_fn, "LDAC_OBJECTS")
    detection_table = pd.DataFrame(np.array(detection_table).byteswap().newbyteorder())

    # add (ra, dec) to DETECTION_TABLE
    w = WCS(filename)
    wcs = []
    for c in detection_table.iterrows():
        wcs.append(w.all_pix2world(c[1]['X_IMAGE']-1, c[1]['Y_IMAGE']-1, 0))
    wcs = pd.DataFrame(wcs, columns=['ALPHA_J2000','DELTA_J2000'])
    final_det_tab = wcs.join(detection_table).astype('float')

    # photometric calibration
    hdr = getheader(filename, 'IMAGE')
    final_det_tab['mag'] = hdr['CALAP']*final_det_tab['MAG_AUTO']+hdr['CALZP']
    

    # add extension table to input FITS
    m = Table(final_det_tab.values, names=final_det_tab.columns)
    hdu = fits.table_to_hdu(m)
    with fits.open(filename, mode='update') as hdul0:
        hdul0.append(hdu)
        hdul0[-1].header['EXTNAME'] = 'DETECTION_TABLE'
        hdul0.flush()
    
    # remove intermediate file
    cmd = ' '.join(['rm', '-rf', intermediate_fn])
    os.system(cmd)


def get_stamp(filename, label, thresh='1.5'):
    """
    To run SExtractor on the image FITS. Searching for all location of the detections and cut the 21x21 thumbnails around the 
    detections. Creating a feature table using the 441 pixel values. 

    Parameters:
    ----------
    filename: str
        filename of the image FITS
    thresh: str of int
        DETECT_THRESH of SExtractor

    Return:
    ----------
    output: str
        filename of the output
    """
    if label == 'bogus':
        image_extract(filename, image_type='DIFFERENCE')
        fn = '_'.join(['DIFFERENCE', filename])
    elif label == 'real':
        fn = filename

    # read in the pixel values from the image
    pix_val = getdata(fn, 'IMAGE')

    # create HDU 'DETECTION_TABLE' if it does not exist
    try:
        df = getdata(fn, 'DETECTION_TABLE')
        df = pd.DataFrame(np.array(df).byteswap().newbyteorder())
    except:
        run_sex(fn, thresh=thresh, detect_minarea='5',deblend_nthresh='16', deblend_mincount='0.01')
        df = getdata(fn, 'DETECTION_TABLE')
        df = pd.DataFrame(np.array(df).byteswap().newbyteorder())

    # filter out flagged detections
    df = df[df.FLAGS==0]
    df = df[df.ERRCXYWIN_IMAGE!=0]

    # get all detections positions
    coor = df[['X_IMAGE','Y_IMAGE']]

    # define stamp columns
    col = ['p'+str(i+1) for i in np.arange(441)] + ['x', 'y']

    # crop stamps for all detections
    feature_tab = [list(Cutout2D(pix_val, (coor.iloc[i]['X_IMAGE']-1, coor.iloc[i]['Y_IMAGE']-1), 
                        (21, 21), mode='partial').data.reshape(441)) + [coor.iloc[i]['X_IMAGE'], coor.iloc[i]['Y_IMAGE']] for i in np.arange(coor.shape[0])]
    
    # fill 0.00001 for pixels outside the edges
    feature_tab = pd.DataFrame(feature_tab, columns=col).fillna(0.00001)
    # create columns for filename and coordinates for re-building purpose
    feature_tab['filename'] = fn

    if label == 'real':
        feature_tab['target'] = 1
    elif label == 'bogus':
        feature_tab['target'] = 0

    feature_tab.dropna(inplace=True)

    output_name = filename.split(".")[0]+'_' + label + '.stp'
    feature_tab.to_csv(output_name, index=False)

    return output_name

