import time
from .image_process import unzip, template_align, image_subtract
from .features import SExtractor, FeatureExtract
from .model import CalcGTR
from .postprocess import position_weighting
from .plot import generate_report
from .xmatch import XmatchGLADE
import pandas as pd


def main(science, template=None, thresh=0.5, xmatch=True, glade=None, report=False):
    # start timer
    start = time.time()

    # funpack image
    unzip(science)

    if not template:
        SExtractor(science, image_ext='IMAGE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        SExtractor(science, image_ext='DIFFERENCE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        diff_features = FeatureExtract(science, 'DIFFERENCE')
        diff_features.make_X()
        diff_features.make_PCA()
        CalcGTR(science, model='RF')
        position_weighting(science)
        if xmatch:
            XmatchGLADE(science, glade, GTR_thresh=thresh)
        if report:
            generate_report(science, thresh=thresh)

    else:
        template = template_align(science, template)
        image_subtract(science, template)
        SExtractor(science, image_ext='IMAGE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        SExtractor(science, image_ext='DIFFERENCE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        diff_features = FeatureExtract(science, 'DIFFERENCE')
        diff_features.make_X()
        diff_features.make_PCA()
        CalcGTR(science, model='RF')
        position_weighting(science)
        if xmatch:
            XmatchGLADE(science, glade, GTR_thresh=xmatch_thresh)
        if report:
            generate_report(science, thresh=thresh)

    end = time.time()
    time_used = end - start
    print('Time elapsed: {}'.format(time_used))