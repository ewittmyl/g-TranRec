import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from .features import fits2df, FitsOp
from astropy.io.fits import getheader

def gauss_weight(filename, sep, n_sigma=1):
    """
    This function is to calculate the exponential weight for weighting the GTR score according to the separation between the 
    detection on the difference image and the closest detection on the science image.

    Parameters
    ----------
    filename: str
        FITS filename you want to operate on.
    sep: list
        Separation in the unit of arcsec between the detection on the difference image and the closest detection on the science image.
    n_sigma: int, optional
        Number of sigma as the decay constant of the exponential function.
    """

    ang_resol = 1.24
    sci_df = fits2df(filename, "PHOTOMETRY")
    sigma = sci_df['FWHM_IMAGE'].mean() * ang_resol
    w = np.exp(-(sep/(n_sigma*sigma)))
    return w

def position_weighting(filename):
    sci_df = fits2df(filename, "PHOTOMETRY")
    diff_df = fits2df(filename, "DIFFERENCE_DETAB")
    diff_coord = SkyCoord(ra=(diff_df['ra']*u.degree).values, dec=(diff_df['dec']*u.degree).values)
    sci_coord = SkyCoord(ra=(sci_df['ra']*u.degree).values, dec=(sci_df['dec']*u.degree).values)
    _, d2d, _ = diff_coord.match_to_catalog_sky(sci_coord)
    d2d = Angle(d2d, u.arcsec).arcsec
    weight = gauss_weight(filename, d2d)
    print("Calculating weighted GTR score...")
    
    diff_df.GTR_score = weight * diff_df.GTR_score
    FitsOp(filename, extname="DIFFERENCE_DETAB", dataframe=diff_df, mode="update")