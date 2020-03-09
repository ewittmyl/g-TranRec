from urllib.request import urlretrieve
import time
import sys
import sqlite3
import pkg_resources
import os
from . import config
import pandas as pd
import numpy as np



glade_url = "http://glade.elte.hu/GLADE_2.3.txt"

def reporthook(count, block_size, total_size):
    # accessary function for showing the downloading progress
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

class GladeDB():

    @staticmethod
    def read_glade(filepath):
        import pandas as pd
        import numpy as np
        
        col = ['PGC','GWGC name','HyperLEDA name',
            '2MASS name','SDSS-DR12 name','flag1',
            'RA','dec','dist','dist_err','z','B',
            'B_err','B_Abs','J','J_err','H','H_err',
            'K','K_err','flag2','flag3']
        print("Loading GLADE catalog...")
        catalog = pd.DataFrame(np.genfromtxt(filepath), columns=col)
        useful_col = ['RA','dec','dist','dist_err','z','B','B_err',
                      'B_Abs','J','J_err','H','H_err','K','K_err']
        return catalog[useful_col]

    @staticmethod
    def create_db():
        if not os.path.isfile("GLADE.txt"):
            print("Downloading GLADE galaxy catalog ...")
            urlretrieve(glade_url, "GLADE.txt", reporthook)

        data_dir = getattr(config, 'DATA_DIR')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        db_path = os.path.join(data_dir, 'glade.db')
        
        gladedf = GladeDB.read_glade("GLADE.txt")

        conn = sqlite3.connect(db_path) # You can create a new database by changing the name within the quotes
        c = conn.cursor() # The database will be saved in the location where your 'py' file is saved
        print("Creating database for GLADE catalog...")
        gladedf.to_sql('glade', conn, if_exists='replace', index = False)

        conn.commit()

    @staticmethod
    def query(cmd):
        data_dir = getattr(config, 'DATA_DIR')
        db_path = os.path.join(data_dir, 'glade.db')

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        useful_col = ['RA','dec','dist','dist_err','z','B','B_err',
                              'B_Abs','J','J_err','H','H_err','K','K_err']
        c.execute(cmd)
        df = pd.DataFrame(c.fetchall(), columns=useful_col)
        conn.commit()
        return df

    @staticmethod
    def image_search(filename):
        from astropy.io.fits import getheader
        from astropy.wcs import WCS
        import numpy as np
        import math
        
        hdr = getheader(filename, 'IMAGE')
        w = WCS(hdr)
        footprint = w.calc_footprint()
        ra_lo, ra_hi = footprint[:,0].min(), footprint[:,0].max()
        dec_lo, dec_hi = footprint[0,:].min(), footprint[0,:].max()
        
        if ((footprint[1,0] > footprint[0,0]) |
            (footprint[2,0] > footprint[3,0])):
            criteria = """WHERE RA>{} AND RA<{} AND dec>{} AND dec<{}""".format(ra_hi,ra_lo,dec_lo,dec_hi)
            
        else:
            criteria = """WHERE RA>{} AND RA<{} AND dec>{} AND dec<{}""".format(ra_lo,ra_hi,dec_lo,dec_hi)
            
        cmd = "SELECT * FROM glade {}".format(criteria)
            
        gal_df = GladeDB.query(cmd)
            
        return gal_df






