import pandas as pd 
import numpy as np
from astropy.nddata import Cutout2D

class Stamping():
    def __init__(self, stamps_arr):
        self.raw_stamps = stamps_arr

    @classmethod
    def create_stamps(cls, image, photo_df):
        print("Obtaining thumbnail arrays...")
        stamps_arr = np.array([Cutout2D(image, (photo_df.iloc[i]['x']-1, photo_df.iloc[i]['y']-1), 
                                (21, 21), mode='partial').data.reshape((21, 21)) for i in np.arange(photo_df.shape[0])])
        return cls(stamps_arr)

    def clean_stamps(self):
        print("Cleaning thumbnails data...")
        # flatten the stamp array
        self.clean_stamps = self.raw_stamps.reshape(-1, 441)
        # replace all 0 by NaN
        self.clean_stamps[self.clean_stamps == 0] = np.nan
        # calculate the median noise level for each row (detection)
        row_median = np.nanmedian(self.clean_stamps, axis=1)
        # find indicies that you need to replace
        inds = np.where(np.isnan(self.clean_stamps))
        # replace all NaN by the detection median
        self.clean_stamps[inds] = np.take(row_median, inds[0])

    def norm_stamps(self):
        print("Normalizing thumbnails...")
        # flatten the stamp array
        flat_stamps = self.clean_stamps.reshape(-1, 441)
        # calculate p-med(p)
        diff = flat_stamps-np.repeat(np.median(flat_stamps, axis=1)+1e-6, 441).reshape((flat_stamps.shape[0], 441))
        # calculate |p-med(p)|/sigma
        s2n = np.abs(diff)/np.repeat(np.std(flat_stamps, axis=1), 441).reshape((flat_stamps.shape[0], 441))
        self.norm_stamps = np.sign(diff)*np.log10(1+s2n)
        self.norm_stamps = np.nan_to_num(self.norm_stamps, nan=1e-5)

        self.norm_stamps = self.norm_stamps.reshape(-1, 21, 21)