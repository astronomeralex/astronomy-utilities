#this library will sample a luminosity function and return luminosities
#this could be useful for monte carlo sampling

#at this point, i will assume that the luminosity function is a schechter function

from schechter import schechter
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import numpy as np
from numpy.random import rand
from math import log10

def samples(lmin,lmax,samples,lstar,alpha,intsamples=100000,constant=False):
    """
    this is the main function for lum_func_sample
    lmin is the minimum luminosity to be considered
    lmax is the maximum luminosity to be considered
    samples of the number of galaxy luminosities to be returned
    lstar is the lstar value for the schechter function
    alpha is the alpha value for the schechter function
    intsamples is the number of samples used internally for integration
    constant is a boolean used for testing. If its true, it will output
    samples number of galaxies all with lmin luminosity
    """
    
    if constant:
        gal = lmin*np.ones(samples)
        return gal

    loglmin = log10(lmin)
    loglmax =log10(lmax)
    
    #first, i will make an array of luminosities using logspace
    l=np.logspace(loglmin,loglmax,intsamples)

    lf=schechter(l,lstar,alpha)

    #now, i make the cumulative integral of lf
    cum = cumtrapz(lf,l)

    #normalize cum so it goes from 0 to 1
    cum = cum/np.max(cum)

    #now i make the interpolating function
    #note this is backwards to you plug in a number from 0 to 1
    #and get back luminosities
    finterp = interp1d(cum,l[:-1],bounds_error=False)

    r=rand(samples)

    gal = finterp(r)
    return gal
