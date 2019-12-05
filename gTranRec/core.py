import time
from .image_process import *
from .sex import *
from astropy.io.fits import getdata, update
from .model import *
from .xmatch import *
from astropy.table import Table
from .plot import *

def extract_all(filename):
    """
    Extracting TEMPLATE and DIFFERENCE from combined science image.

    Parameters:
    ----------
    filename: str
        filename of the combined science image

    Return:
    ----------
    template_fn: str
        filename of the extracted template
    difference_fn: str
        filename of the extracted difference image
    """
    print("Extracting TEMPLATE and DIFFERENCE from {}".format(filename))

    # extract template
    image_extract(filename, image_type='TEMPLATE')
    
    #extract difference image
    image_extract(filename, image_type='DIFFERENCE')

    template_fn = '_'.join(['TEMPLATE', filename])
    difference_fn = '_'.join(['DIFFERENCE', filename])
    
    return template_fn, difference_fn

def try_sex(filename):
    """
    Try to load the DETECTION_TABLE to check existence. If it does not exist, create
    one by 'run_sex'.

    Parameters:
    filename:
        filename of the template or science image.
    """
    try:
        # check if DETECTION_TABLE has been created already
        det_tab = getdata(filename, 'DETECTION_TABLE')
    except:
        # create DETECTION_TABLE if doesn't exist
        print("Running SExtractor on {} to create DETECTION_TABLE...".format(filename))
        run_sex(filename, thresh='1.5', detect_minarea='7', 
                deblend_nthresh='64', deblend_mincount='0.0001')

def score_filter(filename, cutoff_score=0.5):
    """
    Filtering detections below score threshold.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    cutoff_score: float
        decision boundary of the real-bogus score
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Filtering detections with score below {}".format(cutoff_score))
    mask = det_tab['real_bogus'] >= cutoff_score
    det_tab = det_tab[mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def ws_filter(filename, cutoff_score=0.5):
    """
    Filtering detections below score threshold.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    cutoff_score: float
        decision boundary of the real-bogus score
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Filtering detections with score below {}".format(cutoff_score))
    mask = det_tab['w_real_bogus'] >= cutoff_score
    det_tab = det_tab[mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')


def flags_filter(filename):
    """
    Filtering detections with FLAGS.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    # filtering detections with FLAGS do not equal to 0 or 2
    print("Filtering flags...")
    mask = (det_tab['FLAGS'] == 0) | (det_tab['FLAGS'] == 2)
    det_tab = det_tab[mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def calc_mag(filename, science):
    """
    Calculate delta magnitude and magnitude on difference image.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    science: str
        filename of the science image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    # calculate delta magnitude 
    det_tab['del_mag'] = det_tab['mag_temp'] - det_tab['mag_sci']

    # calculate magnitude on difference image
    hdr = getheader(science, 'IMAGE')
    det_tab['mag_diff'] = hdr['CALAP']*det_tab['MAG_AUTO']+hdr['CALZP']

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def artifacts_weight(filename, w=0.2):
    """
    Filtering artifacts.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    # filtering detections look like artifacts on science
    print("Filtering artifacts...")
    mask = ( det_tab['ERRCXYWIN_IMAGE_sci'] == 0 ) | (det_tab['ERRCXYWIN_IMAGE'] > 100) | (det_tab['ERRCXYWIN_IMAGE'] < -100)
    weight = [w if m==1 else 1 for m in mask]
    det_tab['w_real_bogus'] = det_tab['w_real_bogus'] * weight

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def onsci_weight(filename, w=0.2):
    """
    Filtering detections on difference image which cannot be found on 
    the science image.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Weighting off-science detections...")
    mask = det_tab['mag_sci'].isnull()
    weight = [w if m==1 else 1 for m in mask]
    det_tab['w_real_bogus'] = det_tab['real_bogus'] * weight

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def cand_list_operation(filename, det_tab=None, mode='save'):
    """
    Creating CANDIDATES_LIST as an extension table of difference image FITS.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    det_tab: pd.DataFrame / None
        1. DETECTION_TABLE or CANDIDATES_LIST DataFrame
        2. None if using mode='load' 
    mode: str
        ['save','load','update']
    """
    if mode == 'save':
        # create astropy table
        m = Table(det_tab.values, names=det_tab.columns)
        hdu = fits.table_to_hdu(m)
        # add extension table on top of the difference image
        with fits.open(filename, mode='update') as hdul0:
                hdul0.append(hdu)
                hdul0[-1].header['EXTNAME'] = 'CANDIDATES_LIST'
                hdul0.flush()
    
    elif mode == 'load':
        det_tab = getdata(filename, 'CANDIDATES_LIST')
        det_tab = pd.DataFrame(np.array(det_tab).byteswap().newbyteorder())
        return det_tab

    elif mode == 'update':
        m = Table(det_tab.values, names=det_tab.columns)
        hdr = getheader(filename, extname='CANDIDATES_LIST')
        update(filename, np.array(m), extname='CANDIDATES_LIST', header=hdr)

def mp_filter(filename):
    """
    Filtering minor planet.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Filtering Minor Planet...")
    mask = det_tab['mp_offset'] < 10
    det_tab = det_tab[~mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def all_x_ned(filename):
    """
    Xmatch all candidates with NED.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Xmatching with NED catalog...")
    det_tab['NED_obj'] = [xmatch_ned(d[1]['ALPHA_J2000'], d[1]['DELTA_J2000'], r=3) for d in det_tab.iterrows()]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def ned_filter(filename):
    """
    Filtering NED known object.

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print("Filtering NED onject...")
    mask = (det_tab['NED_obj'].isnull()) | (det_tab['NED_obj'] == 3)
    det_tab = det_tab[mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def glade_filter(filename):
    """
    Filtering far glade object

    Parameters:
    ----------
    filename: str
        filename of the difference image
    """
    # load the CANDIDATES_LIST from difference image
    det_tab = cand_list_operation(filename, mode='load')

    print('Filtering detections without galaxy nearby...')
    mask = ~ det_tab.GLADE_offset.isnull()    
    det_tab = det_tab[mask]

    # update the CANDIDATES_LIST from difference image
    cand_list_operation(filename, det_tab=det_tab, mode='update')

def merge_images(difference, science, template):
    """
    Merging template and science image into difference image FITS.

    Parameters:
    ----------
    difference: str
        filename of the difference image FITS
    science: str
        filename of the science image FITS
    template: str
        filename of the template FITS
    """
    with fits.open(difference, mode='update') as hdul0:
        with fits.open(science, memmap=True) as hdul1:
            hdul0.append(hdul1['IMAGE'])
            hdul0.flush()
        with fits.open(template, memmap=True) as hdul1:
            hdul0.append(hdul1['IMAGE'])
            hdul0.flush()
        hdul0[0].header['EXTNAME'] = 'DIFFERENCE'
        hdul0[3].header['EXTNAME'] = 'SCIENCE'
        hdul0[4].header['EXTNAME'] = 'TEMPLATE'
        hdul0.flush()

def main(science, template=None, det_thresh='1.5', algorithm='rf', cutoff_score=0.5, list_by='weighted', xmatch=['mp','ned','glade'], filter_known=['mp','ned'], glade_cat=None, near_galaxy=False, inspect=True):

    # start timer
    start = time.time()

    # uncompress science image
    uncompress(science)

    if not template:
        # extract TEMPLATE and DIFFERENCE if no given input template 
        try:
            template, difference = extract_all(science)
        except:
            raise KeyError("No TEMPLATE or DIFFERENCE in {}!".format(science))
    else:
        # uncompress template image if provided
        uncompress(template)

        # align the template
        template = template_align(science, template)

        # difference imaging
        difference = image_subtract(science, template)

    for img in ([science]+[template]):
        # create DETECTION_TABLE for science and template
        try_sex(img)

    # create DETECTION_TABLE for difference image
    print("Running SExtractor on difference image...")
    run_sex(difference, thresh=det_thresh, detect_minarea='5',deblend_nthresh='16', deblend_mincount='0.01')

    # load model
    model = classifier._load_model(algorithm=algorithm)

    # calculate real-bogus score
    model.predict_all(difference, extname={'image': 'IMAGE', 'table': 'DETECTION_TABLE'})

    # load DETECTION_TABLE from difference image
    det_tab = getdata(difference, 'DETECTION_TABLE')
    det_tab = pd.DataFrame(np.array(det_tab).byteswap().newbyteorder())
    cand_list_operation(difference, det_tab=det_tab, mode='save')

    # fundamental filtering
    score_filter(difference, cutoff_score=cutoff_score)
    flags_filter(difference)

    # xmatch with science image and template
    print("Xmatching with SCIENCE image...")
    xmatch_image(difference, science, suffix='sci', sep=5)
    print("Xmatching with TEMPLATE image...")
    xmatch_image(difference, template, suffix='temp', sep=5)

    # calculate del_mag and mag_diff
    calc_mag(difference, science)

    # filter detections cannot be found on science image if True
    onsci_weight(difference)

    # filtering artifacts
    artifacts_weight(difference)

    if list_by == 'weighted':
        ws_filter(difference, cutoff_score=cutoff_score)
    elif list_by == 'normal':
        pass


    if 'mp' in xmatch:
        try:
            # run PyMPChecker
            mp_check(difference, science, sep=20)
            # reload candidate list
            print("Minor Planet Checking...")
            if 'mp' in filter_known:
                mp_filter(difference)
        except:
            print("No Minor Planet Checker installed!!")

    if 'ned' in xmatch:
        # xmatch the candidates with NED known object
        all_x_ned(difference)
        if 'ned' in filter_known:
            # filter NED known object
            ned_filter(difference)

    if 'glade' in xmatch:
        # xmatch the candidates with GLADE catalog
        if isinstance(glade_cat, pd.DataFrame):
            print('Cross-matching with GLADE catalog...')
            xmatch_glade(difference, glade_cat, sep=20)
        else:
            raise KeyError("Please provide GLADE DataFrame...")
        if near_galaxy:
            glade_filter(difference)
            
    # merge science and template into difference image FITS
    merge_images(difference, science, template)

    # remove all products created by alipy
    os.system("rm -rf ali*")

    # plot all the window images for the candidate list
    if inspect:
        generate_report(difference)

    # show time elapse
    end = time.time()
    time_used = end - start
    print('Time elapsed: {}'.format(time_used))
