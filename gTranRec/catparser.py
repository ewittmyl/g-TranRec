import catsHTM
import pandas as pd
import sys
from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord as skycoord
from math import *
import urllib
import requests	
from . import config

def simbad_check(ra, dec, srad):

	# We use the Simbad script-interface to make our queries.

	# Simbad script-interface URL:
	url = "http://simbad.u-strasbg.fr/simbad/sim-script?script="

	# Our search-script. This can be modified to produce different outputs.
	# Current version will produce each entry in the format:
	# offset (arcsec) | identifier | objtype | RA (deg) | Dec (deg) | U mag | B mag | V mag | R mag | I mag
	script = "output console=off script=off\nformat object form1 \"%DIST | %IDLIST(1) | %OTYPE(S) | %COO(:;A;d) " \
			 "| %COO(:;D;d) | %FLUXLIST(U)[%6.2*(F)] | %FLUXLIST(B)[%6.2*(F)] | %FLUXLIST(V)[%6.2*(F)] " \
			 "| %FLUXLIST(R)[%6.2*(F)] | %FLUXLIST(I)[%6.2*(F)]\"\nquery coo "+str(ra)+" "+str(dec)+" "\
			 +"radius="+str(srad)+"s frame=ICRS"

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

	# Return output, status flag indicating search success:
	return colcell, colunits, out_table, True

def cat_search(ra_in, dec_in, srad):
	ra_in, dec_in = float(ra_in), float(dec_in)


	# give catsHTM rootpath
	catsHTM_rootpath = getattr(config, 'catsHTM_rootpath')


	srad = Angle(float(srad) * u.arcsec)
	cats_srad = srad.arcsec
	# Convert center-position into Astropy SkyCoords:
	position = skycoord(ra=ra_in * u.degree, dec=dec_in * u.degree, frame="icrs")

	def get_center_offset(pos):
		# get separation between two positions
		sep_angle = pos.separation(position)
		return sep_angle.arcsec


	# all catalogs used to check
	all_catalogs = ['AAVSO_VSX', 'TMASS', 'APASS',
								 'GAIADR1', 'GAIADR2', 'IPHAS', 'NEDz', 
								 'IRACgc', 'UCAC4', 'WISE','simbad']



	# define the result table
	useful_col = ['catname','ra','dec','offset']
	all_items_df = pd.DataFrame(columns=useful_col)

	for cat in all_catalogs: # loop through all catalogs
		if cat == 'simbad':
			sb_cols, sb_units, sb_out, sb_success = simbad_check(ra=position.ra.degree, dec=position.dec.degree, srad=srad.arcsec)
			if (sb_success == 1) and (len(sb_out) > 0):
				itemframe = pd.DataFrame(sb_out, columns = sb_cols)
				itemframe['offset'] = [Angle(float(off) * u.arcsec).arcsec for off in itemframe['offset'].values]
				itemframe = itemframe[useful_col[1:]].astype('float')
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
				# change the unit of RA and Dec as degrees
				if colunits['ra'] == 'rad' and colunits['dec'] == 'rad':
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
				# add catalog name
				itemframe['catname'] = cat
				# select useful col only
				itemframe = itemframe[useful_col]
				all_items_df = pd.concat([all_items_df, itemframe], axis=0)
				try:
					if itemframe.offset[0] < 0.5:
						break
				except:
					itemframe
	if all_items_df.empty:
		return all_items_df
	else:
		return all_items_df.sort_values('offset', ascending=1).iloc[[0]]
