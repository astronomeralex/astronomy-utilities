# this modules contains code for various extinction laws
import numpy as np
class ExtinctionCorrector(object):
    """
    Base class for the extinction correctors in this module
    """
    pass



def calzetti_stellar(flux,fluxerr,wave,ebvstellar,ebvlower,ebvupper,Rv = 4.05,Rverr=0.8):
    """
    this will implement the Calzetti law from equation 3
    of Calzetti et al 200, ApJ, 533, 682
    inputs:
        flux, fluxerr -- in consisten units
        wave -- float MICRONS!
        ebvstellar -- E(B-V) for stars in the galaxy
        ebvlower -- lower error bar
        ebvupper -- upper error bar
        Rv -- your favorite Rv -- Calzetti's eqn 5 is assumed
        Rverr
    """
    
    #do sanity checks on waves
    if wave < 0.12 or wave > 2.20:
        print(wave)
        raise Exception("the input wavelength is outside the limits of the Calzetti law")
        
    #now find k'(lambda)
    kl = klambda(wave,Rv)
    
    #find true flux
    trueflux = flux*10**(0.4 * ebvstellar * kl)
    
    #now find the error
    exponentlower = ebvstellar*kl * np.sqrt( (ebvlower / ebvstellar)**2 + (Rverr / kl)**2 )
    exponentupper = ebvstellar*kl * np.sqrt( (ebvupper / ebvstellar)**2 + (Rverr / kl)**2 )
    truefluxlower = trueflux * np.sqrt((fluxerr / flux)**2 + (0.4*np.log(10))**2 * exponentlower**2 )
    truefluxupper = trueflux * np.sqrt((fluxerr / flux)**2 + (0.4*np.log(10))**2 * exponentupper**2 )
    
    #return np.array([trueflux,truefluxlower,truefluxupper])
    return trueflux
        
def klambda(wave,Rv):
    """
    this will calculate k'(lambda) for the Calzetti law
    """
    if wave >= 0.12 and wave < 0.63:
        kl = 2.659*(-2.156 + 1.509 / wave - 0.198 / wave**2 + 0.011 / wave**3) + Rv
    elif wave <= 2.20:
        kl = 2.659*(-1.857 + 1.040 / wave) + Rv
    else:
        raise ArithmeticError(" input can't be larger than 2.2 microns, as the Calzetti law isn't defined there")
        
    return kl
