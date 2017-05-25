"""
======
Cutout
======

Generate a cutout image from a .fits file
"""
from astropy.io import fits
from astropy import wcs
import numpy
import astropy

class DimensionError(ValueError):
    pass

def make_cutouts(fitsfilename, RA, DEC, name, xw, yw, fits_memmap=False):
    """
    fitsfile - fits file name
    RA - array of ras in degrees
    DEC - array of decs in degrees
    name - array of names for output file
    xw,yw - x and y width in pixels
    fits-memmap - should the fits file be memmaped (turn on if large)
    """
    fitswcs = wcs.WCS(fitsfilename)
    fitsfile = fits.open(fitsfilename,memmap = fits_memmap)
    xypix = fitswcs.all_world2pix(RA, DEC, 0)
    for i in  range(len(name)):
        try:
             cutout(fitsfile, xypix[0][i], xypix[1][i], xw, yw, name[i])
        except ValueError:
             print(name[i]," lies outside of image")
    

def cutout(filename, xc, yc, xw, yw, outfile, clobber=True, verbose=False):
    """
    Inputs:
        file  - .fits filename or pyfits HDUList (must be 2D)
        xc,yc - x and y coordinates in the fits files' coordinate system 
        xw,yw - x and y width in pixels 
        outfile - output file name

        #how should i handle boundary cases? fill outside image with zeros?
    """
    if isinstance(filename,str):
        fitsfile = fits.open(filename)
        opened=True
    elif isinstance(filename,astropy.io.fits.hdu.hdulist.HDUList): #check type
        fitsfile = filename
        opened=False
    else:
        raise Exception("cutout: Input file is wrong type (string or HDUList are acceptable).")

    head = fitsfile[0].header.copy()

    if head['NAXIS'] > 2:
        raise DimensionError("Too many (%i) dimensions!" % head['NAXIS'])

    xr = int(xw/2) #distance from center to edge to make below calculation easier
    yr = int(yw/2)

    xmin,xmax = numpy.max([0,xc-xr]),numpy.min([head['NAXIS1'],xc+xr])
    ymin,ymax = numpy.max([0,yc-yr]),numpy.min([head['NAXIS2'],yc+yr])

    if xmax < 0 or ymax < 0:
        raise ValueError("Max Coordinate is outside of map: %f,%f." % (xmax,ymax))
    if ymin >= head.get('NAXIS2') or xmin >= head.get('NAXIS1'):
        raise ValueError("Min Coordinate is outside of map: %f,%f." % (xmin,ymin))

    head['CRPIX1']-=xmin 
    head['CRPIX2']-=ymin
    head['NAXIS1']=int(xmax-xmin)
    head['NAXIS2']=int(ymax-ymin)

    if head.get('NAXIS1') == 0 or head.get('NAXIS2') == 0:
        raise ValueError("Map has a 0 dimension: %i,%i." % (head.get('NAXIS1'),head.get('NAXIS2')))

    img = fitsfile[0].data[ymin:ymax,xmin:xmax]
    newfile = fits.PrimaryHDU(data=img,header=head)
    if verbose: print(("Cut image %s with dims %s to %s.  xrange: %f:%f, yrange: %f:%f" % (filename, file[0].data.shape,img.shape,xmin,xmax,ymin,ymax)))

    newfile.writeto(outfile,clobber=clobber)

    if opened:
        file.close()

