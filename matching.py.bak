#This module will find the closest match in a catalog RA and DEC

import numpy as np
from astropy import coordinates as coord
from astropy import units as u

def matching(objra,objdec,ras,decs,boxsize=1):
    """
    objra and objdec are in degrees
    ras and decs are numpy arrays of all the catalog values (also in degrees)
    """
    obj = coord.ICRSCoordinates(ra=objra, dec=objdec, unit=(u.degree, u.degree))
    distances = []
    for i in xrange(len(ras)):
        if abs(ras[i] - objra) > 0.1 or abs(decs[i] - objdec) > 0.1:
            distances.append(9999)
        else:
            catobj = coord.ICRSCoordinates(ra=ras[i], dec=decs[i], unit=(u.degree, u.degree))
            distances.append(obj.separation(catobj).arcsecs)
    distances = np.array(distances)
    minindex = np.argmin(distances)
    mindist = np.amin(distances)
    b = distances < boxsize
    boxnum = len(distances[b])
    return [minindex,mindist,boxnum]
