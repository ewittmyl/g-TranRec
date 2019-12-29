from .features import fits2df, FitsOp
from astropy.coordinates import Angle
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
import numpy as np
import pandas as pd
from . import config

def read_glade():
    """
    Loading the GLADE catalog from package data directory.

    Return:
    ----------
    cat: pd.DataFrame
        GLADE catalog
    """
    # load GLADE catalog
    GLADE_PATH = getattr(config, 'glade_path')
    col = ['PGC','GWGC name','HyperLEDA name',
            '2MASS name','SDSS-DR12 name','flag1',
            'RA','dec','dist','dist_err','z','B',
            'B_err','B_Abs','J','J_err','H','H_err',
            'K','K_err','flag2','flag3']
    cat = pd.DataFrame(np.genfromtxt(GLADE_PATH), columns=col)
    return cat

def XmatchGLADE(filename, glade_df, GTR_thresh=0.5):
    detab = fits2df(filename, 'DIFFERENCE_DETAB')
    real_df = detab[detab.GTR_score > GTR_thresh]
    bogus_df = detab[detab.GTR_score < GTR_thresh]
    # create prefix for GLADE table columns
    glade_df.columns = 'GLADE_'+glade_df.columns
    
    det_c = SkyCoord(ra=(real_df['ra']*u.degree).values, dec=(real_df['dec']*u.degree).values)
    glade_c = SkyCoord(ra=(glade_df['GLADE_RA']*u.degree).values, dec=(glade_df['GLADE_dec']*u.degree).values)

    idx, d2d, _ = det_c.match_to_catalog_sky(glade_c)
    glade_df = glade_df.iloc[idx,:]

    real_df['GLADE_offset'] = Angle(d2d, u.degree).arcsec
    real_df['GLADE_RA'] = glade_df.GLADE_RA.values
    real_df['GLADE_dec'] = glade_df.GLADE_dec.values
    real_df['GLADE_dist'] = glade_df.GLADE_dist.values

    bogus_df['GLADE_offset'] = np.nan
    bogus_df['GLADE_RA'] = np.nan
    bogus_df['GLADE_dec'] = np.nan
    bogus_df['GLADE_dist'] = np.nan

    detab = real_df.append(bogus_df, ignore_index = True)

    FitsOp(filename, 'DIFFERENCE_DETAB', detab, mode='update')

def mp_check(filename, GTR_thresh=0.5):
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
    detab = fits2df(filename, 'DIFFERENCE_DETAB')
    real_df = detab[detab.GTR_score > GTR_thresh]
    bogus_df = detab[detab.GTR_score < GTR_thresh]

    # image_search with MPChecker
    mpc.image_search(filename, imagetype='IMAGE')
    mp_col = ['RA_deg','Dec_deg']

    try:
        print("Minor Planet Checking...")
        # get coordinates of all mp in the science image
        mp_table = mpc.table[mp_col]

        # create list of coordinates for candidates on difference image
        det_c = SkyCoord(ra=(real_df['ra']*u.degree).values, dec=(real_df['dec']*u.degree).values)

        # create list of coordinates for mp on science image
        mp_coord = SkyCoord(ra=(mp_table['RA_deg']*u.degree).values, dec=(mp_table['Dec_deg']*u.degree).values)

        # xmatching
        idx, d2d, _ = det_c.match_to_catalog_sky(mp_coord)
        
        real_df['mp_offset'] = Angle(d2d, u.degree).arcsec
        bogus_df['mp_offset'] = np.nan

        detab = real_df.append(bogus_df, ignore_index = True)

    except:
        detab['mp_offset'] = np.nan
    
    del mpc
    del mp_table
    
    FitsOp(filename, 'DIFFERENCE_DETAB', detab, mode='update')