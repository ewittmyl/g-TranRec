import os
import pkg_resources

HP_PATH = os.environ['HP_PATH']

sex_cmd = 'sex'
# sex_cmd = 'sextractor'

DATA_DIR = pkg_resources.resource_filename('gTranRec', 'data')

GLADE_PATH = os.path.join(DATA_DIR, 'GLADE.txt')
