import catsHTM
import pandas as pd
import sys
from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord as skycoord
from math import *
import urllib
import requests	
import numpy as np
from . import config
import PyMPChecker as pympc

par = getattr(config, "catsHTM_rootpath")

def simbad_check(ra, dec, srad, galaxy_check=False):
    # We use the Simbad script-interface to make our queries.
    # Simbad script-interface URL:
    url = "http://simbad.u-strasbg.fr/simbad/sim-script?script="

    # Our search-script. This can be modified to produce different outputs.
	# Current version will produce each entry in the format:
	# offset (arcsec) | identifier | objtype | RA (deg) | Dec (deg) | U mag | B mag | V mag | R mag | I mag
	script = "output console=off script=off\nformat object form1 \"%DIST | %IDLIST(1) | %OTYPE(S) | %COO(:;A;d) | %COO(:;D;d) | %FLUXLIST(U)[%6.2*(F)] | %FLUXLIST(B)[%6.2*(F)] | %FLUXLIST(V)[%6.2*(F)] | %FLUXLIST(R)[%6.2*(F)] | %FLUXLIST(I)[%6.2*(F)]\"\nquery coo "+' '.join([str(ra), str(dec)])+" radius="+str(srad)+"s frame=ICRS"

    # Encode the script as URL and build the full search URL-string:
	enc_script = urllib.parse.quote(script)
	search_url = str(url)+str(enc_script)

    # Produce the output-tables (we know what the columns and their units should be; out-table starts
	# empty and we fill it only if sensible results were obtained.
	colcell = ["offset", "identifier", "otype", "ra", "dec", "umag", "bmag", "vmag", "rmag", "imag" ]
	colunits = ["arcsec", "", "", "deg", "deg", "", "", "", "", ""]
	out_table = []

    # Try the search. If it fails, return an empty list and a flag indicating FAILURE, otherwise assume we got a valid table.
	try:
		contents = requests.get(search_url).text	# The actual http-request command for the URL described above
	except:
		return colcell, colunits, out_table, False

    # Parse the results:
	linetable = contents.split("\n")

    for line in linetable:
		# Ignore empty lines
		if len(line) > 0:
			# Check if any object was found; Simbad should return an error-message if there are none.
			if "error" in line:
				break
			elif line == "~~~":
				break
			else:
				linelist = line.split("|")
				out_table.append(linelist)
    
    # convert simbad cross-matched table into dataframe
	df = pd.DataFrame(out_table, columns=colcell)
	# convert dtype of otype as string
	df.otype = df.otype.astype('str')
	df.otype = [d[1:-1] for d in df.otype.values]
	galaxy_label = ['Galaxy','Possible_G','Possible_SClG','Possible_ClG','Possible_GrG','SuperClG','ClG','GroupG','Compact_Gr_G','PairG',
					'IG','PartofG,GinCl','BClG','GinGroup','GinPair','High_z_G','RadioG','HII_G','LSB_G','AGN_Candidate','EmG','StarburstG',
					'BlueComG','AGN','LINER','Seyfert','Seyfert_1','Seyfert_2','Blazar','QSO']
	if galaxy_check:
		# boolean filtering for the rows of galaxies
		galaxy_rows = [1 if v in galaxy_label else 0 for v in df.otype.values]
		galaxy_rows = list(map(bool,galaxy_rows))
		# sort by offset
		df = df[galaxy_rows]
		if not df.empty:
			df = df.sort_values('offset', ascending=1)
    else:
		# boolean filtering for the rows of galaxies
		not_galaxy_rows = [0 if v in galaxy_label else 1 for v in df.otype.values]
		not_galaxy_rows = list(map(bool,not_galaxy_rows))
		# sort by offset
		df = df[not_galaxy_rows]
		if not df.empty:
			df = df.sort_values('offset', ascending=1)
    # redefine colcell and out_table
	colcell = df.columns
	out_table = df.values
	# Return output, status flag indicating search success:
	return colcell, colunits, out_table, True

def cat_search(ra_in, dec_in, srad, conn="gotocompute",galaxy_check=False):
    ra_in, dec_in = float(ra_in), float(dec_in)
    # give catsHTM rootpath
    catsHTM_rootpath = par[conn]
    srad = Angle(float(srad) * u.arcsec)
    cats_srad = srad.arcsec
    # Convert center-position into Astropy SkyCoords:
	position = skycoord(ra=ra_in * u.degree, dec=dec_in * u.degree, frame="icrs")

    def get_center_offset(pos):
		# get separation between two positions
		sep_angle = pos.separation(position)
		return sep_angle.arcsec

    # all catalogs used to check
	all_catalogs = ['TMASS', 'AAVSO_VSX', 'AKARI', 'APASS', 'DECaLS', 'FIRST',
               'GAIADR1', 'GAIADR2', 'IPHAS', 'NEDz', 'PS1', 'PTFpc', 'ROSATfsc',
               'SkyMapper', 'SpecSDSS', 'SAGE', 'IRACgc', 'UCAC4', 'UKIDSS', 'VISTAviking', 'VSTkids','simbad']
	if galaxy_check:
		all_catalogs = ['GLADE','simbad','TMASSxsc','GALEX']
    # define the result table
	useful_col = ['catname','ra','dec','offset','otype']
	all_items_df = pd.DataFrame(columns=useful_col)
    # loop through all catalogs
    for cat in all_catalogs:
        if cat == 'simbad':
            sb_cols, sb_units, sb_out, sb_success = simbad_check(ra=position.ra.degree, dec=position.dec.degree, srad=srad.arcsec, galaxy_check=galaxy_check)
			if (sb_success == 1) and (len(sb_out) > 0):
                itemframe = pd.DataFrame(sb_out, columns = sb_cols)
				itemframe['offset'] = [Angle(float(off) * u.arcsec).arcsec for off in itemframe['offset'].values]
				itemframe = itemframe[useful_col[1:]]
				itemframe[useful_col[1:-1]] = itemframe[useful_col[1:-1]].astype('float')
				itemframe['catname'] = 'simbad'
				itemframe = itemframe[useful_col]
				itemframe = itemframe.sort_values('offset', ascending=1).iloc[[0]]
				all_items_df = pd.concat([all_items_df, itemframe], axis=0)
        else:
            # make sure the catalog name is string
			cat = str(cat)
			# cone search with catsHTM
			cat_out, colcell, colunits = catsHTM.cone_search(cat, position.ra.radian, position.dec.radian,
																 cats_srad, catalogs_dir=catsHTM_rootpath, verbose=False)
            colcell = [colcell[i].lower() for i in range(len(colcell))]
            colunits = [colunits[i].lower() for i in range(len(colunits))]
            # create dict for unit of each column
			colunits = {colcell[i]:colunits[i] for i in range(len(colcell))}

            if len(cat_out) > 0:
                # create empty cross-match result table if the cross-match result is not None
				itemframe = pd.DataFrame(cat_out, columns = colcell)
                if cat == 'NEDz':
					point_mask = itemframe['objtype'] == "*"
					itemframe = itemframe[point_mask]
				if cat == 'IPHAS':
					point_mask = (itemframe['mergedclass'] == -1) & (itemframe['pstar'] > 0.9)
					itemframe = itemframe[point_mask]
				if cat == 'DECaLS':
					point_mask = (itemframe['type'] == 'PSF') & (itemframe['type'] == 'psf')
					itemframe = itemframe[point_mask]
				if cat == 'PS1':
					itemframe['log_likelihood'] = np.log10(np.abs(itemframe['ipsflikelihood'].values))
					point_mask = (itemframe['log_likelihood'] > -3) & (itemframe['log_likelihood'] <= 0) & (itemframe['ipsfmag'] < 19.5)
					itemframe = itemframe[point_mask]
				if cat == 'UCAC4':
					point_mask = (itemframe['flagyale'] == 0) & (itemframe['flagleda'] == 0) & (itemframe['flagextcat'] == 0) & (itemframe['flag2massext'] == 0)
					itemframe = itemframe[point_mask]
				if itemframe.shape[0] > 0:
                    # change the unit of RA and Dec as degrees
					if colunits['ra'] == 'rad' and colunits['dec'] == 'rad':
						itemframe['ra'] = itemframe['ra'].apply(degrees)
						itemframe['dec'] = itemframe['dec'].apply(degrees)
					elif colunits['ra'] == 'radians' and colunits['dec'] == 'radians':
						itemframe['ra'] = itemframe['ra'].apply(degrees)
						itemframe['dec'] = itemframe['dec'].apply(degrees)
					elif colunits['ra'] == 'deg' and colunits['dec'] == 'deg':
						itemframe['ra'] = itemframe['ra'].apply(float)
						itemframe['dec'] = itemframe['dec'].apply(float)
                    # create new column for SkyCoord onjects
					itemframe['skycoord'] = skycoord(itemframe['ra'], itemframe['dec'], unit=u.deg, frame='icrs')
					# get angular difference between the cross-matched objects and the input position
					itemframe['offset'] = itemframe['skycoord'].apply(get_center_offset)
					# sort dataframe by offset
					itemframe = itemframe.sort_values('offset', ascending=1).iloc[[0]]
					# define type of the crossmatched object
					if cat == 'AAVSO_VSX':
						itemframe['otype'] = 'VS'
					elif (cat == 'GLADE') or (cat == 'TMASSxsc') or (cat == 'GALEX') :
						itemframe['otype'] = 'Galaxy'
					else:
						itemframe['otype'] = 'Unknown'
                    # add catalog name
					itemframe['catname'] = cat
					# select useful col only
					itemframe = itemframe[useful_col]
					all_items_df = pd.concat([all_items_df, itemframe], axis=0)

    if all_items_df.empty:
		return all_items_df
	else:
		all_items_df = all_items_df.sort_values('offset', ascending=1)
		all_items_df = all_items_df.reset_index().drop("index",axis=1)
		return all_items_df

def xmatch_df(ra_in, dec_in, srad=10, conn="gotocompute", object_type='point'):
    if  object_type == 'point':
        df = cat_search(ra_in, dec_in, srad, conn=conn, galaxy_check=False)
    elif object_type == 'extend':
        df = cat_search(ra_in, dec_in, srad, conn=conn, galaxy_check=True)
    return df.set_index('catname')

def xmatch_check(photometry_df, srad=10, thresh=0.85, conn="gotocompute"):
    ng_col = ['GAIADR2', 'PS1', 'UCAC4',
       'TMASS', 'AAVSO_VSX', 'APASS', 'DECaLS',
       'IPHAS', 'NEDz','simbad'] # cat col
    photometry_df['known_offset'] = np.nan
    for r in photometry_df.iterrows():
        if r[1].gtr_wcnn > thresh:
            try:
                xm_df = catchecker.xmatch_df(float(r[1]['ra']), float(r[1]['dec']), srad=srad, conn=conn, object_type='point')
                photometry_df.at[r[0], 'known_offset'] = xm_df.iloc[0]['offset']
            except:    
                pass

    g_col = ['simbad', 'GLADE', 'TMASSxsc','GALEX']
    photometry_df['galaxy_offset'] = np.nan
    for r in photometry_df.iterrows():
        if r[1].gtr_wcnn > thresh:
            try:
                xm_df = catchecker.xmatch_df(float(r[1]['ra']), float(r[1]['dec']), srad=60, conn=conn, object_type='extend')
                photometry_df.at[r[0], 'galaxy_offset'] = xm_df.iloc[0]['offset']
            except:    
                pass

    photometry_df['mp_offset'] = np.nan
    mpc = pympc.Checker()
    i = 1
    for r in photometry_df.iterrows():
        if (r[1].gtr_wcnn > thresh) & not (r[1].known_offset<5):
            c1 = SkyCoord(r[1]['ra']*u.degree, r[1]['dec']*u.degree, frame='icrs')
            if i % 10:
                mpc = pympc.Checker()
            mpc.cone_search(r[1]['ra'], r[1]['dec'],r[1]['obsdate'], srad, online=False)
            i += 1
            if mpc.table.shape[0] > 0:
                mp_c = SkyCoord(ra=mpc.table['RA_deg']*u.degree, dec=mpc.table['Dec_deg']*u.degree)
                photometry_df.at[r[0], 'mp_offset'] = round(np.min(c1.separation(mp_c).arcsec), 2)
    i = 1
    for r in photometry_df.iterrows():
        if (r[1].gtr_wcnn > thresh) & not (r[1].known_offset<5) & not (r[1]['mp_offset'] < 5):
            c1 = SkyCoord(r[1]['ra']*u.degree, r[1]['dec']*u.degree, frame='icrs')
            if i % 10:
                mpc = pympc.Checker()
            mpc.cone_search(r[1]['ra'], r[1]['dec'],r[1]['obsdate'], srad, online=True)
            i += 1
            if mpc.table.shape[0] > 0:
                mp_c = SkyCoord(ra=mpc.table['RA_deg']*u.degree, dec=mpc.table['Dec_deg']*u.degree)
                photometry_df.at[r[0], 'mp_offset'] = round(np.min(c1.separation(mp_c).arcsec), 2)

    return photometry_df
