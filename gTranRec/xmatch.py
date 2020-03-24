from .image_process import fits2df, FitsOp
from astropy.coordinates import Angle
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
import numpy as np
import pandas as pd
from . import config
from astroquery.ned import Ned


def XmatchGLADE(detab, glade_df, GTR_thresh=0.5):
    real_df = detab[detab.gtr_score > GTR_thresh]
    bogus_df = detab[detab.gtr_score < GTR_thresh]
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

    return detab

def mp_check(filename, detab, GTR_thresh=0.5):
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
    real_df = detab[detab.gtr_score > GTR_thresh]
    bogus_df = detab[detab.gtr_score < GTR_thresh]

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
    
    return detab

def astroquery_xmatch(detab, r=3, GTR_thresh=0.5):
    real_df = detab[detab.gtr_wscore > GTR_thresh]
    bogus_df = detab[detab.gtr_wscore < GTR_thresh]
    
    coord = SkyCoord(ra=real_df.ra.values, dec=real_df.dec.values, unit=(u.degree, u.degree), frame='icrs')
    r = u.Quantity(r, u.arcsec)
    xmatch_obj = []
    for i in range(real_df.shape[0]):
        c = coord[i]
        ra, dec = real_df.ra.values[i], real_df.dec.values[i]
        print("\Xmatching with NED: {}/{}".format(i+1, real_df.shape[0]), end="\r")
        ned_df = Ned.query_region(c, r_q)
            
        if len(ned_df) == 0:
            j = 0
            while j<5:
                try:
                    print("\Xmatching with SIMBAD: {}/{}".format(i+1, real_df.shape[0]), end="\r")
                    simbad_df = Simbad.query_criteria('region(circle, gal, {0} {1:+f}, {2}s)'.format(ra, dec, r), otype='*')
                    break
                except:
                    simbad_df = None
                    j+=1
            if simbad_df is None:
                xmatch_obj.append(0)
            else:
                xmatch_obj.append(2)
            
        else:
            ned_df.sort('Separation')
            if 'G' in str(ned_df['Type'][0])[2:-1]:
                xmatch_obj.append(0)
            else:
                xmatch_obj.append(1)
    
    real_df['ned_obj'] = xmatch_obj
    bogus_df['ned_obj'] = np.nan
    
    detab = real_df.append(bogus_df, ignore_index = True)
    
    return detab