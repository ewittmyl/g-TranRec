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

    real_df['GLADE_offset'] = d2d
    real_df['GLADE_RA'] = glade_df.GLADE_RA
    real_df['GLADE_dec'] = glade_df.GLADE_dec
    real_df['GLADE_dist'] = glade_df.GLADE_dist

    bogus_df['GLADE_offset'] = np.nan
    bogus_df['GLADE_RA'] = np.nan
    bogus_df['GLADE_dec'] = np.nan
    bogus_df['GLADE_dist'] = np.nan

    detab = real_df.append(bogus_df, ignore_index = True)

    FitsOp(filename, 'DIFFERENCE_DETAB', detab, mode='update')