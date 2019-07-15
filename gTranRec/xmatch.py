import numpy as np
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u
from . import config
from astropy.coordinates import SkyCoord, Distance
from astroquery.ned import Ned
import pkg_resources
from astropy.io.fits import getdata, update, getheader
from astropy.table import Table
import os

def cand_list_operation(filename, det_tab=None, mode='save'):
    """
    Creating CANDIDATES_LIST as an extension table of difference image FITS.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    det_tab: pd.DataFrame / None
        1. DETECTION_TABLE or CANDIDATES_LIST DataFrame
        2. None if using mode='load' 
    mode: str
        ['save','load','update']
    """
    if mode == 'save':
        # create astropy table
        m = Table(det_tab.values, names=det_tab.columns)
        hdu = fits.table_to_hdu(m)
        # add extension table on top of the difference image
        with fits.open(filename, mode='update') as hdul0:
                hdul0.append(hdu)
                hdul0[-1].header['EXTNAME'] = 'CANDIDATES_LIST'
                hdul0.flush()
    
    elif mode == 'load':
        det_tab = getdata(filename, 'CANDIDATES_LIST')
        det_tab = pd.DataFrame(np.array(det_tab).byteswap().newbyteorder())
        return det_tab

    elif mode == 'update':
        m = Table(det_tab.values, names=det_tab.columns)
        hdr = getheader(filename, extname='CANDIDATES_LIST')
        update(filename, np.array(m), extname='CANDIDATES_LIST', header=hdr)

def xmatch_image(difference, image, suffix, sep=5):
    """
    Xmatching DETECTION_TABLE on the difference image with template or science image. 

    Parameters:
    ----------
    difference: str
        filename of difference image
    image: str
        filename of the FITS image for xmatching
    suffix: str
        suffix for new columns
    sep: float
        searching size of xmatching in arcsec
    """
    # define sep in degree
    sep_deg = Angle(sep, u.arcsec).degree

    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(difference, mode='load')

    # load DETECTION_TABLE from the image FITS
    cat = getdata(image, 'DETECTION_TABLE')
    cat = pd.DataFrame(np.array(cat).byteswap().newbyteorder())
    cat.columns = cat.columns + '_' + suffix

    # create list of coordinates for detections on difference image
    diff_c = SkyCoord(ra=(det_tab['ALPHA_J2000']*u.degree).values, dec=(det_tab['DELTA_J2000']*u.degree).values)

    # create list of coordinates for detections on xmatched image
    cat_c = SkyCoord(ra=(cat['ALPHA_J2000_'+suffix]*u.degree).values, dec=(cat['DELTA_J2000_'+suffix]*u.degree).values)

    # xmatching the above two coordinate lists
    idx, d2d, d3d = diff_c.match_to_catalog_sky(cat_c)
    cat = cat.iloc[idx,:]
    selected_col = [col+'_'+suffix for col in ['mag', 'MAGERR_AUTO', 'ERRCXYWIN_IMAGE']]
    cat = cat[selected_col]
    cat['angDist'] = d2d
    cat = cat.reset_index().drop("index",axis=1)

    # filter un-xmatched results with angDist > sep
    mask = cat['angDist'] > sep_deg
    cat[mask] = np.nan

    # merge table
    mtab = det_tab.join(cat[selected_col])

    # update CANDIDATES_LIST
    cand_list_operation(difference, det_tab=mtab, mode='update')

def mp_check(filename, science_image, sep=20):
    """
    Cross-match with the Minor Planet Catalog within the FoV of 
    the image within a given region.

    Parameters
    ----------
    filename: str 
        filename of the difference image
    science_image: str
        filename of the science image
    sep: float
        searching region in arcsec

    Return
    ----------
    marged cross-matched candidate list
    """

    
    import PyMPChecker as pympc
    mpc = pympc.Checker()


    # read CANDIDATES_LIST
    cand_list = cand_list_operation(filename, mode='load')

    # image_search with MPChecker
    mpc = pympc.Checker()   
    mpc.image_search(science_image, imagetype='IMAGE')
    mp_col = ['RA_deg','Dec_deg']

    try:
        # get coordinates of all mp in the science image
        mp_table = mpc.table[mp_col]

        # create list of coordinates for candidates on difference image
        cand_coord = SkyCoord(ra=(cand_list['ALPHA_J2000']*u.degree).values, dec=(cand_list['DELTA_J2000']*u.degree).values)

        # create list of coordinates for mp on science image
        mp_coord = SkyCoord(ra=(mp_table['RA_deg']*u.degree).values, dec=(mp_table['Dec_deg']*u.degree).values)

        # xmatching
        idx, d2d, d3d = cand_coord.match_to_catalog_sky(mp_coord)
        mp_table = mp_table.iloc[idx,:]
        mp_table['mp_offset'] = d2d
        mp_table = mp_table.reset_index().drop("index",axis=1)
        mp_table['mp_offset'] = Angle(mp_table['mp_offset'].values, u.degree).arcsec

        # filter mp with offset > searching size
        mask = mp_table['mp_offset'] > sep
        mp_table[mask] = np.nan
        mtab = cand_list.join(mp_table['mp_offset'])

    except:
        # empty mp table on the FoV of science image
        mp_table = pd.DataFrame(np.nan, index=np.arange(cand_list.shape[0]), columns=mp_col+['mp_offset'])
        mtab = cand_list.join(mp_table['mp_offset'])

    cand_list_operation(filename, det_tab=mtab, mode='update')

def classification(obj_type):
    """
    Conversion dictionary of object class.

    Parameters:
    ----------
    obj_type: str
        object type indicated by NED catalog

    Return:
    ----------
    integer indicating which class of object
    """
    # define the dictionary
    conversion = {'Flare*':1,
                '!Flar*':1,
                'V*':2,
                '!V*':2,
                'G':3,
                'SN':4,
                '!SN':4,
                '*':5}
    try:
        return conversion[obj_type]
    except KeyError:
        return 0

def xmatch_ned(ra, dec, r=3):
    """
    Cross-match with the NED catalog with given cone of search.

    Parameters:
    ----------
    ra: float
        ra in degree
    dec: float
        dec in degree
    r: float
        searching radius in arcsec 

    Return:
    ----------
    integer indicating which class of object
    """
    # create SkyCoord object for the (ra,dec)
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    # define searching radius in arcsec
    radius = u.Quantity(r, u.arcsec)

    # cross-match using astroquery
    tab = Ned.query_region(coord, radius)
    try:
        # pick the closest one within the search cone
        tab.sort('Separation')
        obj_type = [str(d['Type'])[2:-1] for d in tab]

        # return the object class
        return classification(obj_type[0])
    except:
        # return NaN if no object within the search cone
        return np.nan

def read_glade():
    """
    Loading the GLADE catalog from package data directory.

    Return:
    ----------
    cat: pd.DataFrame
        GLADE catalog
    """
    # load GLADE catalog
    GLADE_PATH = getattr(config, 'GLADE_PATH')
    col = ['PGC','GWGC name','HyperLEDA name',
            '2MASS name','SDSS-DR12 name','flag1',
            'RA','dec','dist','dist_err','z','B',
            'B_err','B_Abs','J','J_err','H','H_err',
            'K','K_err','flag2','flag3']
    cat = pd.DataFrame(np.genfromtxt(GLADE_PATH), columns=col)
    return cat

def xmatch_glade(filename, cat, sep=20):    
    """
    Xmatch with the GLADE catalog with given searching radius.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    cat: pd.DataFrame
        GLADE catalog in pd.DataFrame
    sep: float
        search radius in arcsec
    """

    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')
    
    # create prefix for GLADE table columns
    cat.columns = 'GLADE_'+cat.columns

    # create list of coordinates for detections on difference image
    diff_c = SkyCoord(ra=(det_tab['ALPHA_J2000']*u.degree).values, dec=(det_tab['DELTA_J2000']*u.degree).values)

    # create list of coordinates for detections on xmatched image
    cat_c = SkyCoord(ra=(cat['GLADE_RA']*u.degree).values, dec=(cat['GLADE_dec']*u.degree).values)

    # xmatching the above two coordinate lists
    idx, d2d, d3d = diff_c.match_to_catalog_sky(cat_c)
    cat = cat.iloc[idx,:]
    cat['GLADE_offset'] = d2d
    cat = cat.reset_index().drop("index",axis=1)
    cat['GLADE_offset'] = Angle(cat['GLADE_offset'].values, u.degree).arcsec

    # filter un-xmatched results with offset > sep
    mask = cat['GLADE_offset'] > sep
    cat[mask] = np.nan

    # merge table
    selected_col = ['GLADE_RA','GLADE_dec','GLADE_dist','GLADE_offset']
    mtab = det_tab.join(cat[selected_col])

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=mtab, mode='update')