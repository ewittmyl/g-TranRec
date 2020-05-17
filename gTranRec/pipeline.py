import time
from .image_process import unzip, template_align, image_subtract, fits2df, FitsOp
from .features import SExtractor
from .plot import generate_report
import pandas as pd
from .cnn import CNN
from .weighting import Weighting
from .xmatch import all_Xmatch
import os

def add_score(filename):
    param = {
        'sciphoto': 'PHOTOMETRY'
    }
    # calculate CNN score
    c = CNN.load()
    c.image_predict(filename)
    # calculate weighting
    sciphoto = fits2df(filename, param['sciphoto'])
    w = Weighting(sciphoto, c.photo_df)
    w.calc_weight()
    w.diffphoto['gtr_wcnn'] = w.diffphoto['weight'] * w.diffphoto['gtr_cnn']

    return w.diffphoto

def main(science, template=None, thresh=0.69, near_galaxy=False, report=False):
    # start timer
    start = time.time()

    # funpack image
    unzip(science)

    if not template:
        diffphoto = add_score(science)
        diffphoto = all_Xmatch(science, diffphoto, thresh=thresh)
        diffphoto.drop(columns=diffphoto.columns[diffphoto.dtypes=='object'], inplace=True)

        FitsOp(science, "PHOTOMETRY_DIFF", diffphoto, mode="update")

        if report:
            generate_report(science, thresh=thresh, near_galaxy=near_galaxy)

    else:
        unzip(template)
        template = template_align(science, template)
        image_subtract(science, template)
        os.system("rm -rf {}".format(template))
        SExtractor(science, image_ext='IMAGE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        SExtractor(science, image_ext='DIFFERENCE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        diffphoto = add_score(science)
        diffphoto = all_Xmatch(science, diffphoto, thresh=thresh)
        diffphoto.drop(columns=diffphoto.columns[diffphoto.dtypes=='object'], inplace=True)
        FitsOp(science, "PHOTOMETRY_DIFF", diffphoto, mode="update")
        if report:
            generate_report(science, output='{}_report_sub.pdf'.format(science.split('.')[0]), thresh=thresh, near_galaxy=near_galaxy)

    end = time.time()
    time_used = end - start
    print('Time elapsed: {}'.format(time_used))