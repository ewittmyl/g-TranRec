import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from .image_process import fits2df, FitsOp
from astropy.io.fits import getheader


class CalcWeight():
    def __init__(self, filename, detab):
        self.filename = filename
        self.detab = detab

    def onsci_weight(self, onsci_sep, n_sigma=1):
        ang_resol = 1.24
        sci_df = fits2df(self.filename, "IMAGE_DETAB")
        self.sigma = sci_df['FWHM_IMAGE'].mean() * ang_resol
        self.onsci_w = np.exp(-(onsci_sep/(n_sigma*self.sigma)))
    
    
    def overall_weight(self, gal_sep, n_sigma=4):
        self.overall_w = (1-self.onsci_w)*np.exp(-gal_sep/(n_sigma*self.sigma)) + self.onsci_w
    

    def calculate(self):
        sci_df = fits2df(self.filename, "IMAGE_DETAB")
        diff_coord = SkyCoord(ra=(self.detab['ra']*u.degree).values, dec=(self.detab['dec']*u.degree).values)
        sci_coord = SkyCoord(ra=(sci_df['ra']*u.degree).values, dec=(sci_df['dec']*u.degree).values)
        _, d2d, _ = diff_coord.match_to_catalog_sky(sci_coord)
        d2d = Angle(d2d, u.arcsec).arcsec
        self.onsci_weight(d2d)
        gal_sep = np.nan_to_num(self.detab.GLADE_offset.values, nan=100)
        self.overall_weight(gal_sep)
        self.detab['weight'] = self.overall_w
        return self.detab

