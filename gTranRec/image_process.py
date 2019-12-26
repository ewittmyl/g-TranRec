from astropy.io.fits import getdata, getheader
from astropy.io import fits
import numpy as np
import pandas as pd
import alipy, os


def unzip(filename):
    """
    To funpack the given file.

    Parameters
    ----------
    filename : str
        String of the filename needed to be unzipped.
    """
    print('Funpacking {}...'.format(filename))

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
    print("Appending TEMPLATE into {}...".format(science))
    with fits.open(science, mode='update') as hdul0:
        with fits.open(output, memmap=True) as hdul1:
            hdul0.append(hdul1[0])
            hdul0.flush()
        hdul0[-1].header['EXTNAME'] = 'TEMPLATE'
        hdul0.flush()
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
    output = '_'.join(['diff', science])
    try:
        hpcmd = 'hotpants -inim ' + science + '[0] -tmplim ' + aligned_template + ' -outim ' + output + ' -tu 55000 -iu 55000 -tl 10 -il 10 -tg 1.3 -ig 1.3 -tr 10 -ir 10 -nrx 2 -nry 2 -fi 0 -n i -ko 2 -sconv -bgo 2 -v 0 -ng 3 6 1.428 4 2.855 2 5.710 -r 19.985 -rss 25.985'
        os.system(hpcmd)
    except:
        # return error if no hotpants is found
        raise FileNotFoundError('Error occurs when running HOTPANTS!') 

    print("Appending DIFFERENCE into {}...".format(science))
    with fits.open(science, mode='update') as hdul0:
        with fits.open(output, memmap=True) as hdul1:
            hdul0.append(hdul1[0])
            hdul0.flush()
        hdul0[-1].header['EXTNAME'] = 'DIFFERENCE'
        hdul0.flush()

    os.system("rm -rf {}".format(aligned_template))
    os.system("rm -rf {}".format(output))



def image_extract(filename, extname='IMAGE'):
        """
        Extracting a particular image from the FITS file.

        Parameters
        ----------
        filename: str
            String of the image filename.
        image_type: str 
            [IMAGE / SCIENCE / TEMPLATE / DIFFERENCE]
        """
        print("Extracting '{}' from {}...".format(extname, filename))
        # read image
        data = getdata(filename, extname)

        # read header
        hdr = getheader(filename, extname)

        # change extension name
        hdr['EXTNAME'] = 'IMAGE'

        # re-create new FITS for extracted image
        hdu = fits.PrimaryHDU(data, hdr)
        output = '_'.join([filename.split(".")[0], extname+'.fits'])
        hdu.writeto(output, clobber=True)

# def scale_stamp(stamp):
#         print("Normalizing thumbnails...")
#         # flatten the stamp array
#         flat_stamps = stamp.ravel()
#         # calculate p-med(p)
#         diff = flat_stamps-(np.median(flat_stamps)+1e-6)
#         # calculate |p-med(p)|/sigma
#         s2n = np.abs(diff)/np.std(flat_stamps)
#         norm_stamps = np.sign(diff)*np.log10(1+s2n)
#         norm_stamps = norm_stamps.reshape(21, 21)
#         return norm_stamps
