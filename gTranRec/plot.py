import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.table import Table
from astropy.io.fits import getdata, getheader
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from fpdf import FPDF
import os
from .image_process import fits2df

def generate_report(filename, output=None, thresh=0.85):
        detab = fits2df(filename, 'PHOTOMETRY_DIFF')
        photo_sci = fits2df(filename, "PHOTOMETRY")
        # select detections with score > thresh
        candidates = detab[detab.gtr_wcnn>thresh]
        # round off the data values for cleaner display
        candidates = round(candidates, 5)
        # sort table by score
        candidates = candidates.sort_values(by='gtr_wcnn', ascending=False)
        # filter out edge detections
        edge_filter = (candidates.X_IMAGE > 50) & (candidates.Y_IMAGE > 50) & (candidates.X_IMAGE < 8126) & (candidates.Y_IMAGE < 6082)
        candidates = candidates[edge_filter]
        # filter out too large FWHM
        fwhm_cutoff = photo_sci.FWHM_IMAGE.quantile(0.95)
        candidates = candidates[candidates.FWHM_IMAGE < fwhm_cutoff]
        # filter out abnormal mag
        hdr = getheader(filename, "IMAGE")
        limmag = hdr['CALLIM5']
        candidates = candidates[candidates.mag < limmag]
        candidates = candidates[candidates.mag > 14]
        # filter known source and keep sources next to galaxy
        # mp_filter = candidates.mp_offset < 8
        # candidates = candidates[~mp_filter]
        # known_filter = candidates.known_offset < 5
        # galaxy_filter  = candidates.galaxy_offset < 60
        # candidates = candidates[(~known_filter) | (galaxy_filter)]
      

        pix_val = []
        pix_val.append(getdata(filename, 'IMAGE'))
        pix_val.append(getdata(filename, 'TEMPLATE'))
        pix_val.append(getdata(filename, 'DIFFERENCE'))
        col = ['ra','dec','X_IMAGE','Y_IMAGE', 'gtr_wcnn','mag','galaxy_offset','known_offset','mp_offset']

        candidates = candidates[col]
       
        
        interval = ZScaleInterval()
        j = 0
        stamps_fn = []
        candidates = candidates.replace(np.nan, "--", regex=True)
        for candidate in candidates.iterrows():
                fig = plt.figure(figsize=(10,10))
                for i, img_type in enumerate(['IMAGE','TEMPLATE','DIFFERENCE']):
                        stamp = [Cutout2D(pix_val[i], (float(candidate[1]['X_IMAGE'])-1, float(candidate[1]['Y_IMAGE'])-1), (150, 150),  mode='partial').data.reshape(22500)]
                        stamp = interval(stamp)
                        img = stamp[0].reshape(150,150)
                        ax = plt.subplot(1, 3, i+1)
                        ax.imshow(img, cmap="gray")
                        if i == 0:
                                ax.set_xticks([0,16.1])
                                ax.set_xticklabels(['','20"'])
                                if ('galaxy_offset' in candidates.columns) & (candidate[1]['galaxy_offset'] != '--'):
                                        r = float(candidate[1]['galaxy_offset']) / 1.24
                                        if r < 75:
                                                circle = plt.Circle((75, 75), r, color='r', fill=False, linewidth=0.8)
                                                ax.add_artist(circle)
                        else:
                                ax.set_xticks([])
                                ax.set_xticklabels([])
                        ax.set_yticks([])

                        ax.axhline(75, xmin=0.4, xmax=0.45,color='red', linewidth=2)
                        ax.axhline(75, xmin=0.55, xmax=0.6,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.4, ymax=0.45,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.55, ymax=0.6,color='red', linewidth=2)
                
                        if i == 0:
                                plt.title("{}\nRA: {}\nDec: {}\nMagnitude: {}\nScore: {}".format(filename, candidate[1]['ra'], candidate[1]['dec'], candidate[1]['mag'], candidate[1]['gtr_wcnn']), loc='left', fontsize=10)
                        # if i == 1:
                        #         plt.title("Known Off: {}\nGalaxy Off: {}\nMP Off: {}".format(candidate[1]['known_offset'], candidate[1]['galaxy_offset'], candidate[1]['mp_offset']), loc='left', fontsize=10)    
                image_name = filename.split(".")[0] + '_' + str(j) + '.png'
                stamps_fn.append(image_name)
                plt.savefig(image_name, dpi=100, bbox_inches='tight')
                plt.close()
                j += 1
        ##############################################
        pdf = FPDF(orientation = 'L')
        for i in range(len(stamps_fn)):
                print("\rProcessing: {}/{}".format(i+1, len(stamps_fn)), end="\r")
                pdf.add_page()
                pdf.image(filename.split(".")[0] + '_' + str(i) + '.png')
        if output is None:
                report_fn = filename.split(".")[0] + '_report.pdf'
        else:
                report_fn = output
        pdf.output(report_fn, "F")
        os.system("rm -rf *png")
