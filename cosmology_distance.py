#this library will provide different cosmological measures of distance
#these routines are based off formulae from
#Hogg 1999 Distance Measures in Cosmology arXiv:astro-ph/9905116v4
#Written by Alex Hagen (hagen@psu.edu)

import scipy.constants as constants
import numpy as np
import scipy.integrate as integrate

def distcalc(z,h=0.70,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    this function will calculate the distance measures and will output a dictionary
    that can be used by other functions
    the inputs are the redshift, z, you want to calculate the distance to,
    omegalambda, the dimensionless density parameter for dark matter
    omegam, the dimensionless density parameter for matter
    omegak, the dimensionless density parameter for curvature
    outputs distances in meters unless otherwise noted
    """

    H0 = 100 * h # this is in units of km/s/Mpc

    H0freq = H0 * constants.kilo/(constants.mega * constants.parsec) # this is H0 is units of Hz
    
    hubbletime = 1.0/H0freq # in seconds
    hubbletimeyr = hubbletime / constants.year

    #hubble distance
    dh = constants.c / H0freq # in meters

    #now i can calculate the comoving distance (line of sight) using hogg eqn 15
    dc = dh * integrate.quad(dcintegrand,0,z,(omegalambda,omegam,omegak))[0]

    #now i can find the transverse comoving distance using hogg eqn 16
    if omegak == 0:
        dm = dc
    elif omegak > 0:
        dm = dh/np.sqrt(omegak) * np.sinh(dc * np.sqrt(omegak) / dh)
    else:
        dm = dh/np.sqrt(abs(omegak)) * np.sinh(dc * np.sqrt(abs(omegak)) / dh)


    #now i will calculate the angular diameter distance (hogg eqn 18)
    da = dm/(1+z)
    
    #now i will calculate scale in kpc/arcsec, since this is commonly used
    scale = da * constants.arcsec / (constants.kilo * constants.parsec)

    #now i will calculate the luminosity distance (hog eqn 21)
    dl = (1+z)*dm
    
    #now i will calculate lookback time and 
    #time from the begining of the universe to that redshift using hogg eqn 30
    
    tlookback = hubbletimeyr * integrate.quad(timeintegrand,0,z,(omegalambda,omegam,omegak))[0]
    
    tz = hubbletimeyr * integrate.quad(timeintegrand,z,np.inf,(omegalambda,omegam,omegak))[0]

    #for output, i will make a dictionary
    output = dict(dh=dh,dc=dc,dm=dm,da=da,scale=scale,dl=dl,tlookback = tlookback,tz=tz)

    return output
    
def timeintegrand(z,omegalambda,omegam,omegak):
    """
    this helper function will return the integrand needed
    for lookback time calculations
    """

    return 1./((1+z)*adotovera(z,omegalambda,omegam,omegak))

def adotovera(z,omegalambda,omegam,omegak):
    """
    this helper function will calculate \dot{a} / a, which is needed to
    find the comoving distance
    """
    return np.sqrt(omegam*(1+z)**3 + omegak*(1+z)**2 + omegalambda)

def dcintegrand(z,omegalambda,omegam,omegak):
    """
    this will calculate the integrand needed for finding the comoving distance
    """
    return 1./adotovera(z,omegalambda,omegam,omegak)
    
    
def dh(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns hubble distance in meters
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['dh']

def dc(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns line of sight comoving distance in meters
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['dc']

def dm(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns tranverse comoving distance in meters
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['dm']

def da(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns angular diameter distance in meters
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['da']
    
def scale(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns scale in kpc/arcsec
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['scale']

def dl(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):
    """
    returns luminosity distance in meters
    """
    return distcalc(z,h,omegalambda,omegam,omegak)['dl']
    
def tlookback(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):

    return distcalc(z,h,omegalambda,omegam,omegak)['tlookback']
    
def tz(z,h=0.7,omegalambda=0.7,omegam=0.3,omegak=0.0):

    return distcalc(z,h,omegalambda,omegam,omegak)['tz']    
