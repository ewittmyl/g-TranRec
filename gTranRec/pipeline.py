import time
from .image_process import unzip, template_align, image_subtract, fits2df, FitsOp
from .sex import SExtractor
from .plot import generate_report
import pandas as pd
from .cnn import CNN
from .weighting import Weighting
import os
from .catsHTM_check import xmatch_check

def add_score(filename):
    param = {
        'sciphoto': 'PHOTOMETRY'
    }
    # load model
    c = CNN.load()
    # predict
    c.image_predict(filename)
    # read PHOTOMETRY_SCIENCE
    sciphoto = fits2df(filename, param['sciphoto'])
    # calculate weight according to the separation between the detections on the science and the difference images
    w = Weighting(sciphoto, c.photo_df)
    w.calc_weight()
    # calculate the weighted CNN score
    w.diffphoto['gtr_wcnn'] = w.diffphoto['weight'] * w.diffphoto['gtr_cnn']
    # return PHOTOMETRY_DIFFERENCE with CNN score as pd.DataFrame
    return w.diffphoto

def main(science, template=None, thresh=0.85, report=True):
    # start timer
    start = time.time()
    # funpack image
    unzip(science)
    # use the default difference image if input template is not given
    if not template:
        # add CNN score onto PHOTOMETRY_DIFFERENCE and return the updated one
        diffphoto = add_score(science)
        
        ### diffphoto = all_Xmatch(science, diffphoto, thresh=thresh)

        # drop all columns with object dtypes
        diffphoto.drop(columns=diffphoto.columns[diffphoto.dtypes=='object'], inplace=True)
        # update the new PHOTOMETRY_DIFFERENCE
        FitsOp(science, "PHOTOMETRY_DIFF", diffphoto, mode="update")
        # generate report PDF if report=True
        if report:
            generate_report(science, thresh=thresh)
    # perform image subtraction if template is given
    else:
        # unzip template image
        unzip(template)
        # template image alignment
        template = template_align(science, template)
        # image subtraction
        image_subtract(science, template)
        # remove template
        os.system("rm -rf {}".format(template))
        # run SExtractor on science image 
        SExtractor(science, image_ext='IMAGE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        # run SExtractor on template
        SExtractor(science, image_ext='DIFFERENCE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        # add CNN score onto PHOTOMETRY_DIFFERENCE and return the updated one
        diffphoto = add_score(science)
        
        ### diffphoto = all_Xmatch(science, diffphoto, thresh=thresh)

         # drop all columns with object dtypes
        diffphoto.drop(columns=diffphoto.columns[diffphoto.dtypes=='object'], inplace=True)
        # update the new PHOTOMETRY_DIFFERENCE
        FitsOp(science, "PHOTOMETRY_DIFF", diffphoto, mode="update")
        # generate report PDF if report=True
        if report:
            generate_report(science, output='{}_report_sub.pdf'.format(science.split('.')[0]), thresh=thresh)

    # end the timer
    end = time.time()
    # calculate time consumed on the script
    time_used = end - start
    # display the time consumption
    print('Time elapsed: {}'.format(time_used))