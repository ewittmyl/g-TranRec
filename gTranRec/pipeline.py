import time
from .image_process import unzip, template_align, image_subtract
from .features import SExtractor, FeatureExtract
from .model import CalcGTR
from .postprocess import position_weighting


def main(science, template=None, xmatch_thresh=0.5, glade=None):
    # start timer
    start = time.time()

    # funpack image
    unzip(science)

    if not template:
        SExtractor(science, image_ext='DIFFERENCE').run()
        diff_features = FeatureExtract(science, 'DIFFERENCE')
        diff_features.make_X()
        diff_features.make_PCA()
        CalcGTR(science, model='RF')
        position_weighting(science)
        if glade:
            XmatchGLADE(science, glade, xmatch_thresh=xmatch_thresh)

    else:
        template = template_align(science, template)
        image_subtract(science, template)
        SExtractor(science, image_ext='DIFFERENCE').run()
        diff_features = FeatureExtract(science, 'DIFFERENCE')
        diff_features.make_X()
        diff_features.make_PCA()
        CalcGTR(science, model='RF')
        position_weighting(science)
        if glade:
            XmatchGLADE(science, glade, xmatch_thresh=xmatch_thresh)

    end = time.time()
    time_used = end - start
    print('Time elapsed: {}'.format(time_used))