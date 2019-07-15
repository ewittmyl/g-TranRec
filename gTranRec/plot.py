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

def generate_report(image, img_types=['SCIENCE','TEMPLATE','DIFFERENCE']):
        cand_list = pd.DataFrame(np.array(getdata(image, 'CANDIDATES_LIST')).byteswap().newbyteorder())
        cand_list = cand_list.sort_values(by='real_bogus', ascending=False)
        cand_list = round(cand_list,3)
        cand_list = cand_list.replace(np.nan, "--", regex=True)
        pix_val = []
        pix_val.append(getdata(image, img_types[0]))
        pix_val.append(getdata(image, img_types[1]))
        pix_val.append(getdata(image, img_types[2]))
        col = ['ALPHA_J2000','DELTA_J2000','X_IMAGE','Y_IMAGE', 'real_bogus','mag_sci','del_mag','mag_diff']
        if 'mp_offset' in cand_list.columns:
                col += ['mp_offset']
        if 'NED_obj' in cand_list.columns:
                col += ['NED_obj']
        if 'GLADE_offset' in cand_list.columns:
                col += ['GLADE_offset']

        coords = cand_list[col]
        interval = ZScaleInterval()
        j = 0
        for coord in coords.iterrows():
                fig = plt.figure(figsize=(8,8))
                for i, img_type in enumerate(img_types):
                        win = [Cutout2D(pix_val[i], (float(coord[1]['X_IMAGE'])-1, float(coord[1]['Y_IMAGE'])-1), (150, 150),  mode='partial').data.reshape(22500)]
                        win = interval(win)
                        img = win[0].reshape(150,150)
                        ax = plt.subplot(1, 3, i+1)
                        ax.imshow(img, cmap="gray")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.axhline(75, xmin=0.4, xmax=0.45,color='red', linewidth=2)
                        ax.axhline(75, xmin=0.55, xmax=0.6,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.4, ymax=0.45,color='red', linewidth=2)
                        ax.axvline(75, ymin=0.55, ymax=0.6,color='red', linewidth=2)
                
                        if i == 0:
                                plt.title("{}\nRA: {}\nDec: {}\nScore: {}".format(image, coord[1]['ALPHA_J2000'], coord[1]['DELTA_J2000'], coord[1]['real_bogus']), loc='left', fontsize=10)
                        if i == 1:
                                plt.title("Science m: {}\nDelta m: {}\nDiff m: {}".format(coord[1]['mag_sci'],coord[1]['del_mag'],coord[1]['mag_diff']), loc='left', fontsize=10)
                        if i == 2:
                                if ('mp_offset' in coords.columns) & ('NED_obj' in coords.columns) & ('GLADE_offset' in coords.columns):
                                        plt.title("Minor Planet: {}\nNED Object: {}\nGLADE: {}".format(coord[1]['mp_offset'], coord[1]['NED_obj'], coord[1]['GLADE_offset']), loc='left', fontsize=10)
                                elif ('mp_offset' in coords.columns) & ('NED_obj' in coords.columns):
                                        plt.title("Minor Planet: {}\nNED Object: {}".format(coord[1]['mp_offset'], coord[1]['NED_obj']), loc='left', fontsize=10)
                                elif ('NED_obj' in coords.columns) & ('GLADE_offset' in coords.columns):
                                        plt.title("NED Object: {}\nGLADE: {}".format(coord[1]['NED_obj'], coord[1]['GLADE_offset']), loc='left', fontsize=10)
                                elif ('mp_offset' in coords.columns) & ('GLADE_offset' in coords.columns):
                                        plt.title("Minor Planet: {}\nGLADE: {}".format(coord[1]['mp_offset'], coord[1]['GLADE_offset']), loc='left', fontsize=10)
                                elif 'mp_offset' in coords.columns:
                                        plt.title("Minor Planet: {}".format(coord[1]['mp_offset']), loc='left', fontsize=10)
                                elif 'NED_obj' in coords.columns:
                                        plt.title("NED Object: {}".format(coord[1]['NED_obj']), loc='left', fontsize=10)
                                elif 'GLADE_offset' in coords.columns:
                                        plt.title("GLADE: {}".format(coord[1]['GLADE_offset']), loc='left', fontsize=10)
                                else:
                                        continue                
                image_name = image.split("_")[1]+image.split("_")[2].split("-")[0] + '_' + str(j) + '.png'
                plt.savefig(image_name, dpi=100, bbox_inches='tight')
                plt.close()
                j += 1
        ##############################################
        pdf = FPDF(orientation = 'L')
        thumbnail_list = [image.split("_")[1]+image.split("_")[2].split("-")[0] + '_' + str(i) + '.png' for i in np.arange(0,j)]
        thumbnail_list.sort()
        for i, thumbnail in enumerate(thumbnail_list):
                print("\rProcessing: {}/{}".format(i+1, len(thumbnail_list)), end="\r")
                pdf.add_page()
                pdf.image(thumbnail)
        report_fn = "_".join([image.split("_")[1], image.split("_")[2].split("-")[0],'report.pdf'])
        pdf.output(report_fn, "F")
        os.system("rm -rf *png")