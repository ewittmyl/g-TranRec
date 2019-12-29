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
from .features import fits2df

def generate_report(filename, thresh=0.5):
        detab = fits2df(filename, 'DIFFERENCE_DETAB')
        candidates = detab[detab.GTR_score>thresh]
        candidates = candidates.sort_values(by='GTR_score', ascending=False)
        candidates = round(candidates, 5)
        candidates = candidates.replace(np.nan, "--", regex=True)

        pix_val = []
        pix_val.append(getdata(filename, 'IMAGE'))
        pix_val.append(getdata(filename, 'TEMPLATE'))
        pix_val.append(getdata(filename, 'DIFFERENCE'))
        col = ['ra','dec','X_IMAGE','Y_IMAGE', 'GTR_score','mag']

        # if 'mp_offset' in cand_list.columns:
        #         col += ['mp_offset']
        # if 'NED_obj' in cand_list.columns:
        #         col += ['NED_obj']
        # if 'GLADE_offset' in cand_list.columns:
        #         col += ['GLADE_offset']
        # if 'GLADE_dist' in cand_list.columns:
        #         col += ['GLADE_dist']

        candidates = candidates[col]
        interval = ZScaleInterval()
        j = 0
        stamps_fn = []
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
                        else:
                                ax.set_xticks([])
                                ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.axhline(75, xmin=0.4, xmax=0.45,color='red', linewidth=2)
                        ax.axhline(75, xmin=0.55, xmax=0.6,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.4, ymax=0.45,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.55, ymax=0.6,color='red', linewidth=2)
                
                        if i == 0:
                                plt.title("{}\nRA: {}\nDec: {}\nScore: {}".format(filename, candidate[1]['ra'], candidate[1]['dec'], candidate[1]['GTR_score']), loc='left', fontsize=10)
                        if i == 1:
                                plt.title("Magnitude: {}".format(candidate[1]['mag']), loc='left', fontsize=10)
                        if i == 2:
                                if 'GLADE_offset' in candidates.columns:
                                        print(candidate[1]['GLADE_offset'])
                                        plt.title("GLADE: {}'', {}Mpc".format(candidate[1]['GLADE_offset'], candidate[1]['GLADE_dist']), loc='left', fontsize=10)
                        #         if ('mp_offset' in coords.columns) & ('NED_obj' in coords.columns) & ('GLADE_offset' in coords.columns):
                        #                 plt.title("Minor Planet: {}\nNED Object: {}\nGLADE: {}'', {}Mpc".format(coord[1]['mp_offset'], coord[1]['NED_obj'], coord[1]['GLADE_offset'], coord[1]['GLADE_dist']), loc='left', fontsize=10)
                        #         elif ('mp_offset' in coords.columns) & ('NED_obj' in coords.columns):
                        #                 plt.title("Minor Planet: {}\nNED Object: {}".format(coord[1]['mp_offset'], coord[1]['NED_obj']), loc='left', fontsize=10)
                        #         elif ('NED_obj' in coords.columns) & ('GLADE_offset' in coords.columns):
                        #                 plt.title("NED Object: {}\nGLADE: {}'', {}Mpc".format(coord[1]['NED_obj'], coord[1]['GLADE_offset'], coord[1]['GLADE_dist']), loc='left', fontsize=10)
                        #         elif ('mp_offset' in coords.columns) & ('GLADE_offset' in coords.columns):
                        #                 plt.title("Minor Planet: {}\nGLADE: {}'', {}Mpc".format(coord[1]['mp_offset'], coord[1]['GLADE_offset'], coord[1]['GLADE_dist']), loc='left', fontsize=10)
                        #         elif 'mp_offset' in coords.columns:
                        #                 plt.title("Minor Planet: {}".format(coord[1]['mp_offset']), loc='left', fontsize=10)
                        #         elif 'NED_obj' in coords.columns:
                        #                 plt.title("NED Object: {}".format(coord[1]['NED_obj']), loc='left', fontsize=10)
                        #         elif 'GLADE_offset' in coords.columns:
                        #                 plt.title("GLADE: {}'', {}Mpc".format(coord[1]['GLADE_offset'], coord[1]['GLADE_dist']), loc='left', fontsize=10)
                        #         else:
                        #                 continue                
                image_name = filename.split(".")[0] + '_' + str(j) + '.png'
                stamps_fn.append(image_name)
                plt.savefig(image_name, dpi=100, bbox_inches='tight')
                plt.close()
                j += 1
        ##############################################
        pdf = FPDF(orientation = 'L')
        stamps_fn.sort()
        for i, thumbnail in enumerate(stamps_fn):
                print("\rProcessing: {}/{}".format(i+1, len(stamps_fn)), end="\r")
                pdf.add_page()
                pdf.image(thumbnail)
        report_fn = filename.split(".")[0] + '_report.pdf'
        pdf.output(report_fn, "F")
        os.system("rm -rf *png")
