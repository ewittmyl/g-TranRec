from astropy.io.fits import getdata, getheader, update
from astropy.io import fits
import numpy as np
import pandas as pd
import alipy, os
from astropy.table import Table

def FitsOp(filename, extname, dataframe, mode='append'): 
    # append extra extension table into FITS if mode=append
    if mode == 'append': 
        # display progress if verbose=True
        print("Adding new extension table {} into {}...".format(extname, filename))
        # convert pd.DataFrame to astropy table
        m = Table(dataframe.values, names=dataframe.columns)
        # create header unit
        hdu = fits.table_to_hdu(m)
        # open FITS
        with fits.open(filename, mode='update') as hdul0:
            # append extra extension
            hdul0.append(hdu)
            # rename the extension table
            hdul0[-1].header['EXTNAME'] = extname
            # update changes to the FITS
            hdul0.flush()
    # create new FITS if mode=write
    elif mode == 'write':
        # display progress if verbose=True
        print("Creating new FITS {}...".format(filename))
        # convert pd.DataFrame to astropy table
        astropy_tab = Table.from_pandas(dataframe)
        # create new FITS
        astropy_tab.write(filename, format='fits')
        # open FITS
        with fits.open(filename, mode='update') as hdul0:
            # rename the extension table
            hdul0[-1].header['EXTNAME'] = extname
            # update changes to the FITS
            hdul0.flush()
    # update particular extension table if mode=update
    elif mode == 'update':
        # display progress if verbose=True
        print("Updating {}[{}]...".format(filename, extname))
        # convert pd.DataFrame to astropy table
        m = Table(dataframe.values, names=dataframe.columns)
        # read header for the extention table
        hdr = getheader(filename, extname=extname)
        # update the extention table 
        update(filename, np.array(m), extname=extname, header=hdr)
    print("Done!")


def fits2df(filename, extname):
    # display progress if verbose=True
    print("Reading {} in {}...".format(extname, filename))
    # get data from the extention table in the FITS
    df = getdata(filename, extname)
    # convert the table to pd.DataFrame
    df = pd.DataFrame(np.array(df).byteswap().newbyteorder())
    # return extension table in pd.DataFrame
    print("Done!")
    return df


def unzip(filename):
    # display progress if verbose=True
    print('Funpacking {}...'.format(filename))
    # rename file with suffix '.fz'
    output = ''.join([filename, '.fz'])
    # create unix command (rename)
    cmd = ' '.join(['mv', filename, output])
    # run unix command
    os.system(cmd)
    # create unix command (funpack)
    cmd = ' '.join(['funpack', output])
    # run unix command
    os.system(cmd)
    # create unix command (remove)
    cmd = ' '.join(['rm', '-rf', output])
    # run unix command
    os.system(cmd)
    print("Done!")


def template_align(science, template):
    # display progress if verbose=True
    print("Aligning {} to match with {}...".format(template, science))
    # list of the images needed to be aligned
    ali_img_list = [template]
    # try hdu=0
    try:
        # define hdu=0
        hdu = 0
        # identify field stars to xmatch
        identifications = alipy.ident.run(science, ali_img_list, hdu=hdu, visu=False)
        # list of the same length as ali_img_list
        for id in identifications:
            if id.ok == True: # i.e., if it worked
                # display progress if verbose=True
                print("%20s : %20s, flux ratio %.2f" % (id.ukn.name, id.trans, id.medfluxratio))

            else:
                # display progress if verbose=True
                print("%20s : no transformation found !" % (id.ukn.name))
        # define the shape of the aligned image
        outputshape = alipy.align.shape(science, hdu=hdu)
        # creating the aligned image
        output = '_'.join(['aligned', template])
        # perform alignment
        alipy.align.affineremap(template, id.trans,  
                                outputshape, hdu=0, alifilepath=output)
    # try hdu=1 if hdu=0 create error
    except:
        # define hdu=1
        hdu = 1
        # identify field stars to xmatch
        identifications = alipy.ident.run(science, ali_img_list, hdu=hdu, visu=False)
        # list of the same length as ali_img_list
        for id in identifications:
            if id.ok == True: # i.e., if it worked
                # display progress if verbose=True
                print("%20s : %20s, flux ratio %.2f" % (id.ukn.name, id.trans, id.medfluxratio))
            else:
                # display progress if verbose=True
                print("%20s : no transformation found !" % (id.ukn.name))
        # define the shape of the aligned image
        outputshape = alipy.align.shape(science, hdu=hdu)
        # creating the aligned image
        output = '_'.join(['aligned', template])
        # perform alignment
        alipy.align.affineremap(template, id.trans,  
                                outputshape, hdu=0, alifilepath=output)
    # display progress if verbose=True
    print("Image Alignment completed. {} is created.".format(output))
    print("Copying science image to a new FITS...")
    # rename SCIECNE image
    os.system("mv {} {}.copy".format(science, science))
    # extract IMAGE from the FITS
    image_extract('.'.join([science, 'copy']), output=science, extname='IMAGE')
    # remove the backup SCIENCE image
    os.system("rm -rf {}.copy".format(science))
    # display progress if verbose=True
    print("Appending TEMPLATE into {}...".format(science))
    # open FITS
    with fits.open(science, mode='update') as hdul0:
        # open aligned image
        with fits.open(output, memmap=True) as hdul1:
            # append the aligned image
            hdul0.append(hdul1[0])
            hdul0.flush()
        # append the template
        hdul0[-1].header['EXTNAME'] = 'TEMPLATE'
        hdul0.flush()
    # return the filename of the aligned template
    print("Done!")
    return output


def image_subtract(science, aligned_template):
    # display progress if verbose=True
    print("Running {} - {}".format(science, aligned_template))
    # define output name of the difference image
    output = '_'.join(['diff', science])
    # try to perform image subtraction
    try:
        # define hotpants command
        hpcmd = 'hotpants -inim ' + science + '[0] -tmplim ' + aligned_template + ' -outim ' + output + ' -tu 55000 -iu 55000 -tl 10 -il 10 -tg 1.3 -ig 1.3 -tr 10 -ir 10 -nrx 2 -nry 2 -fi 0 -n i -ko 2 -sconv -bgo 2 -v 1 -ng 3 6 1.428 4 2.855 2 5.710 -r 19.985 -rss 25.985'
        # run hotpants command
        os.system(hpcmd)
    except:
        # return error if no hotpants is found
        raise FileNotFoundError('Error occurs when running HOTPANTS!') 
    # display progress if if verbose=True
    print("Appending DIFFERENCE into {}...".format(science))
    # open FITS
    with fits.open(science, mode='update') as hdul0:
        # open subtracted image
        with fits.open(output, memmap=True) as hdul1:
            # append subtracted image as DIFFERENCE
            hdul0.append(hdul1[0])
            # save change
            hdul0.flush()
        # rename DIFFERENCE
        hdul0[-1].header['EXTNAME'] = 'DIFFERENCE'
        hdul0.flush()
    # remove all by-products
    os.system("rm -rf {}".format(aligned_template))
    os.system("rm -rf {}".format(output))
    os.system("rm -rf alipy_out")
    print("Done!")




def image_extract(filename, output=None, extname='IMAGE'):
    # display progress if verbose=True
    print("Extracting '{}' from {}...".format(extname, filename))
    # read the given image
    data = getdata(filename, extname)
    # read the header of the given image
    hdr = getheader(filename, extname)
    # change extension name
    hdr['EXTNAME'] = 'IMAGE'
    # re-create new FITS for extracted image
    hdu = fits.PrimaryHDU(data, hdr)
    # create output FITS if the output name is provided
    if not output:
        output = '_'.join([filename.split(".")[0], extname+'.fits'])
    hdu.writeto(output, clobber=True)
    print("Done!")
