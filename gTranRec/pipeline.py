import time
from .image_process import unzip, template_align, image_subtract
from .features import SExtractor, CalcALL
from .model import CalcGTR
from .plot import generate_report
from .xmatch import XmatchGLADE, mp_check
import pandas as pd
import os


def main(science, template=None, thresh=0.5, near_galaxy=False, report=False):
    # start timer
    start = time.time()

    # funpack image
    unzip(science)

    if not template:
        c = CalcALL(science)
        c.run(thresh=thresh)
        if report:
            generate_report(science, thresh=thresh, near_galaxy=near_galaxy)

    else:
        unzip(template)
        template = template_align(science, template)
        image_subtract(science, template)
        os.system("rm -rf {}".format(template))
        SExtractor(science, image_ext='IMAGE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        SExtractor(science, image_ext='DIFFERENCE').run(thresh=2, deblend_nthresh=32, deblend_mincont=0.005)
        c = CalcALL(science)
        c.run(thresh=thresh)
        if report:
            generate_report(science, output='{}_report_sub.pdf'.format(science.split('.')[0]), thresh=thresh, near_galaxy=near_galaxy)

    end = time.time()
    time_used = end - start
    print('Time elapsed: {}'.format(time_used))