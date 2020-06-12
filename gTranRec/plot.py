import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.table import Table
from astropy.io.fits import getdata
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from fpdf import FPDF
import os
from .image_process import fits2df

def generate_report(filename, output=None, thresh=0.85, near_galaxy=True, xmatch_filter=True):
        detab = fits2df(filename, 'PHOTOMETRY_DIFF')
        candidates = detab[detab.gtr_cnn>thresh]
        candidates = round(candidates, 5)
        candidates = candidates.sort_values(by='gtr_cnn', ascending=False)

        pix_val = []
        pix_val.append(getdata(filename, 'IMAGE'))
        pix_val.append(getdata(filename, 'TEMPLATE'))
        pix_val.append(getdata(filename, 'DIFFERENCE'))
        col = ['ra','dec','X_IMAGE','Y_IMAGE', 'gtr_cnn','mag']

        if 'mp_offset' in candidates.columns:
                col += ['mp_offset']
                candidates = candidates[candidates.mp_offset>5]
        if 'GLADE_offset' in candidates.columns:
                col += ['GLADE_offset','GLADE_RA','GLADE_dec']
        if 'GLADE_dist' in candidates.columns:
                col += ['GLADE_dist']
        if 'known_ra' in candidates.columns:
                col += ['known_ra','known_dec','known_off']

        candidates = candidates[col]
        if near_galaxy:
                candidates = candidates[candidates.GLADE_offset<30]
        if xmatch_filter and ('known_off' in candidates.columns):
                candidates = candidates[np.isnan(candidates.known_off)]
        interval = ZScaleInterval()
        j = 0
        stamps_fn = []
        candidates = candidates.replace(np.nan, "--", regex=True)
        for candidate in candidates.iterrows():
                fig = plt.figure(figsize=(8,8))
                for i, img_type in enumerate(['IMAGE','TEMPLATE','DIFFERENCE']):
                        stamp = [Cutout2D(pix_val[i], (float(candidate[1]['X_IMAGE'])-1, float(candidate[1]['Y_IMAGE'])-1), (150, 150),  mode='partial').data.reshape(22500)]
                        stamp = interval(stamp)
                        img = stamp[0].reshape(150,150)
                        ax = plt.subplot(1, 3, i+1)
                        ax.imshow(img, cmap="gray")
                        if i == 0:
                                ax.set_xticks([0,16.1])
                                ax.set_xticklabels(['','20"'])
                                if 'GLADE_offset' in candidates.columns:
                                        r = candidate[1]['GLADE_offset'] / 1.24
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
                                plt.title("{}\nRA: {}\nDec: {}\nScore: {}".format(filename, candidate[1]['ra'], candidate[1]['dec'], candidate[1]['gtr_cnn']), loc='left', fontsize=10)
                        if i == 1:
                                plt.title("Magnitude: {}".format(candidate[1]['mag']), loc='left', fontsize=10)
                                if 'mp_offset' in candidates.columns:
                                        if 'known_ra' in candidates.columns:
                                                plt.title("Magnitude: {}\nMinor Planet: {}''\nCat off: {}".format(candidate[1]['mag'], candidate[1]['mp_offset'], candidate[1]['known_off']), loc='left', fontsize=10)
                                        else:
                                                plt.title("Magnitude: {}\nMinor Planet: {}''".format(candidate[1]['mag'], candidate[1]['mp_offset']), loc='left', fontsize=10)

                        if i == 2:
                                if 'GLADE_offset' in candidates.columns:
                                        plt.title("GLADE galaxy\n{}'', {}Mpc\nRA, Dec: {}, {}".format(candidate[1]['GLADE_offset'], candidate[1]['GLADE_dist'], candidate[1]['GLADE_RA'], candidate[1]['GLADE_dec']), loc='left', fontsize=10)          
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
