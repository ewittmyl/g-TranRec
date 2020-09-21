import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u

class Weighting():
    def __init__(self, sciphoto, diffphoto):
        self.sciphoto = sciphoto
        self.diffphoto = diffphoto

    def calc_sigma(self):
        # dividing image by quadrants
        x_quad_label = [i for i in range(1,7)]
        y_quad_label = [i for i in range(1,5)]
        self.sciphoto['x_quad'] = pd.cut(self.sciphoto.x,[0,1363,2725,4088,5451,6813,8176.1],right=False,labels=x_quad_label)
        self.sciphoto['y_quad'] = pd.cut(self.sciphoto.y,[0,1533,3066,4599,6132.1],right=False,labels=y_quad_label)
        self.sciphoto['med_fwhm'] = self.sciphoto[['x_quad','y_quad','FWHM_IMAGE']].groupby(['x_quad','y_quad']).transform(lambda x: x.median()).FWHM_IMAGE

    def calc_weight(self):
        param = {
            'n_sig': 100, 
            'ang_sol': 1.24,
        }
        self.calc_sigma()
        print("Calculating offset between SCIENCE and DIFFERENCE..")
        # get all detection coordinates for both science and difference photometry tables
        diff_det_coor = SkyCoord(ra=(self.diffphoto['ra'].values*u.degree), dec=(self.diffphoto['dec'].values*u.degree))
        sci_det_coor = SkyCoord(ra=(self.sciphoto['ra'].values*u.degree), dec=(self.sciphoto['dec'].values*u.degree))
        # cross-match two coordinate lists
        idx, d2d, _ = diff_det_coor.match_to_catalog_sky(sci_det_coor)
        # change unit from degree to arcsec
        d2d = Angle(d2d, u.arcsec).arcsec
        # rearrange the photometry table for science image
        xmatch_scidf = self.sciphoto.iloc[idx]

        # define offset and median FWHM
        offset = d2d
        med_fwhm = xmatch_scidf.med_fwhm.values

        # calculate weight
        weight = np.exp(-(offset/(param['n_sig']*med_fwhm)))
        self.diffphoto['weight'] = weight
        print('Done!')
