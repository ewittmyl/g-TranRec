from .image_process import fits2df, FitsOp
from astropy.coordinates import Angle
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
import numpy as np
import pandas as pd
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from astroquery.heasarc import Heasarc
from .database import GladeDB
from .catparser import cat_search


def XmatchGLADE(detab, glade_df, GTR_thresh=0.85):
    real_df = detab[detab.gtr_cnn > GTR_thresh]
    bogus_df = detab[detab.gtr_cnn < GTR_thresh]
    # create prefix for GLADE table columns
    glade_df.columns = 'GLADE_'+glade_df.columns
    
    det_c = SkyCoord(ra=(real_df['ra'].values*u.degree), dec=(real_df['dec'].values*u.degree))
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

    return detab

def mp_check(filename, detab, GTR_thresh=0.85):
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
    real_df = detab[detab.gtr_cnn > GTR_thresh]
    bogus_df = detab[detab.gtr_cnn < GTR_thresh]

    # image_search with MPChecker
    mpc.image_search(filename, imagetype='IMAGE')
    mp_col = ['RA_deg','Dec_deg']

    try:
        print("Minor Planet Checking...")
        # get coordinates of all mp in the science image
        mp_table = mpc.table[mp_col]

        # create list of coordinates for candidates on difference image
        det_c = SkyCoord(ra=(real_df['ra'].values*u.degree), dec=(real_df['dec'].values*u.degree))

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
    
    return detab

def contextual_check(detab, r=1, GTR_thresh=0.85):
    # split detection table into real and bogus subset
    real_df = detab[detab.gtr_cnn > GTR_thresh]
    bogus_df = detab[detab.gtr_cnn < GTR_thresh]
    # define lists for information of the known objects
    known_ra = []
    known_dec = []
    known_off = []
    known_df = pd.DataFrame()

    for c in zip(real_df.ra.values, real_df.dec.values):
        check_df = cat_search(c[0], c[1], r)
        if check_df.shape[0] > 0:
            known_ra.append(check_df.ra.values[0])
            known_dec.append(check_df.dec.values[0])
            known_off.append(check_df.offset.values[0])
        else:
            known_ra.append(np.nan)
            known_dec.append(np.nan)
            known_off.append(np.nan)
    
    real_df['known_ra'] = known_ra
    real_df['known_dec'] = known_ra
    real_df['known_off'] = known_ra

    bogus_df['known_ra'] = np.nan
    bogus_df['known_dec'] = np.nan
    bogus_df['known_off'] = np.nan    

    detab = real_df.append(bogus_df, ignore_index = True)
    return detab
        

def all_Xmatch(filename, diffphoto, thresh=0.85, catparse=True):
    # MP check
    diffphoto = mp_check(filename, diffphoto, thresh)
    if catparse:
        # load sciphoto from FITS
        sciphoto = fits2df(filename, 'PHOTOMETRY')
        diffphoto = contextual_check(diffphoto, GTR_thresh=thresh)
        # except:
        #     print("Cannot X-match with NED and SIMBAD catalog...")

    return diffphoto

