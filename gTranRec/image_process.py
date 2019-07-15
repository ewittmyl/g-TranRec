import alipy, os
from astropy.io.fits import getdata, getheader
from astropy.io import fits
from . import config
import numpy as np
import pandas as pd

def uncompress(filename):
        """
        Uncompressing FITS with funpack.

        Parameters:
        ----------
        filename: str
                filename of the FITS
        """
        print('Uncompressing image: {}'.format(filename))

        # rename file with suffix '.fz'
        output = ''.join([filename, '.fz'])
        cmd = ' '.join(['mv', filename, output])
        os.system(cmd)

        # funpack file
        cmd = ' '.join(['funpack', output])
        os.system(cmd)

        # remove suffixed file
        cmd = ' '.join(['rm', '-rf', output])
        os.system(cmd)

def uncompress_all():
        """
        Uncompressing all FITS in the current directory with funpack.
        """
        # list out all FITS in the current directory
        files = [fn for fn in os.listdir("./") if '.fits' in fn]

        for fn in files:
                # uncompress one-by-one
                uncompress(fn)

def template_align(science, template):
        """
        Aligning the template to match with the FoV of science image and created aligned template.

        Parameters:
        ----------
        science: str
                filename of the science image
        template: str
                filename of the template
        
        Return:
        output: str
                filename of the aligned template
        """
        print("Aligning {} to match with {}...".format(template, science))

        # list of the images needed to be aligned
        ali_img_list = [template]
        
        try:
                hdu = 0
                # identify field stars to xmatch
                identifications = alipy.ident.run(science, ali_img_list, hdu=hdu, visu=False)

                for id in identifications: # list of the same length as ali_img_list
                        if id.ok == True: # i.e., if it worked

                                print("%20s : %20s, flux ratio %.2f" % (id.ukn.name, id.trans, id.medfluxratio))

                        else:
                                print("%20s : no transformation found !" % (id.ukn.name))

                # define the shape of the aligned image
                outputshape = alipy.align.shape(science, hdu=hdu)

                # creating the aligned image
                output = '_'.join(['aligned', template])
                alipy.align.affineremap(template, id.trans,  
                                        outputshape, hdu=0, alifilepath=output)
        except:
                hdu = 1
                # identify field stars to xmatch
                identifications = alipy.ident.run(science, ali_img_list, hdu=hdu, visu=False)

                for id in identifications: # list of the same length as ali_img_list
                        if id.ok == True: # i.e., if it worked

                                print("%20s : %20s, flux ratio %.2f" % (id.ukn.name, id.trans, id.medfluxratio))

                        else:
                                print("%20s : no transformation found !" % (id.ukn.name))

                # define the shape of the aligned image
                outputshape = alipy.align.shape(science, hdu=hdu)

                # creating the aligned image
                output = '_'.join(['aligned', template])
                alipy.align.affineremap(template, id.trans,  
                                        outputshape, hdu=0, alifilepath=output)

        print("Image Alignment completed. {} is created.".format(output))
        return output

def image_subtract(science, aligned_template):
        """
        Subtracting aligned template by science image using HotPants.

        Parameters:
        ----------
        science: str
                filename of the science image
        aligned_template: str
                filename of the aligned template
        
        Return:
        output: str
                filename of the difference image
        """
        print("Running {} - {}".format(science, aligned_template))
        HP_PATH = getattr(config, 'HP_PATH')
        output = '_'.join(['diff', science])
        if os.path.isfile(HP_PATH):
                hpcmd = HP_PATH + ' -inim ' + science + '[0] -tmplim ' + aligned_template + ' -outim ' + output + ' -tu 55000 -iu 55000 -tl 10 -il 10 -tg 1.3 -ig 1.3 -tr 10 -ir 10 -nrx 2 -nry 2 -fi 0 -n i -ko 2 -sconv -bgo 2 -v 0 -ng 3 6 1.428 4 2.855 2 5.710 -r 19.985 -rss 25.985'
                os.system(hpcmd)
                return output
        else:
                # return error if no hotpants is found
                raise FileNotFoundError('No HOTPANTS is found!') 

def image_extract(filename, image_type='IMAGE'):
        """
        Extracting image from FITS.

        Parameters:
        ----------
        filename: str
                filename of the FITS
        image_type: str 
                [IMAGE / SCIENCE / TEMPLATE / DIFFERENCE]
                which image is extracted from the FITS (default 'IMAGE')
        """

        # read image
        data = getdata(filename, image_type)

        # read header
        hdr = getheader(filename, image_type)

        # change extension name
        hdr['EXTNAME'] = 'IMAGE'

        # re-create new FITS for extracted image
        hdu = fits.PrimaryHDU(data, hdr)
        output = '_'.join([image_type, filename.split(".")[0]+'.fits'])
        hdu.writeto(output, clobber=True)
        
        
def scaling(feature_tab):
        """
        Scaling the raw pixel values in the stamp image.

        Parameters:
        ----------
        feature_tab: pd.DataFrame
                DataFrame containing p1 to p441
        
        Return:
        ----------
        scaled_stamps: pd.DataFrame
                scaled input table
        """
        pixel_col = ['p'+str(i+1) for i in np.arange(441)]

        # only use pixel values to be the features
        stamps = feature_tab[pixel_col]

        # define x - median(x) of each stamp (NaN might exist)
        diff = stamps.sub(stamps.median(axis=1)+1e-6, axis='index')

        # create scaled stamp table (NaN might exist)
        scaled_stamps = (diff/np.abs(diff))*np.log10(1+(np.abs(diff).div(stamps.std(axis=1), axis='index'))**1)

        # fill masked pixels with median - sigma
        scaled_stamps = scaled_stamps.T.fillna(scaled_stamps.median(axis=1)).T

        return scaled_stamps