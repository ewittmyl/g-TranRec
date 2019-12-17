import pkg_resources
import os


sex_cmd = 'sextractor'
# sex_cmd = 'sex'

user = 'travis'
password = '19880830'


MODEL_DIR = pkg_resources.resource_filename('gTranRec', 'model')

pca_path = os.path.join(MODEL_DIR, 'pca.m')
rf_path = os.path.join(MODEL_DIR, 'rf.m')