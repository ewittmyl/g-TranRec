import pkg_resources
import os


# sex_cmd = 'sextractor'
sex_cmd = 'sex'

user = 'travis'
password = '19880830'


DATA_DIR = pkg_resources.resource_filename('gTranRec', 'data')

pca_path = os.path.join(DATA_DIR, 'pca.m')
rf_path = os.path.join(DATA_DIR, 'rf.m')
glade_path = os.path.join(DATA_DIR, 'GLADE.txt')

catsHTM_rootpath = {
    'gotocompute':str("/export/gotodata2/catalogs/"),
    'goto2':str("/mnt4/data/ewittmyl/catalogs/")
}
