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

def CSB10_Alambda(wavelength,R,B):
    """
    from appendix of conroy, schminovich, and blanton 2010. ApJ 718 184
    wavelength is in units of microns
    R is E(B-V)/Av
    B is bump strength -- 1.0 for MW bump
    output is Alambda / Av
    """
    #assume x is a float, not array -- should eventually make this work for arrays
    x = 1/wavelength
    assert x > 0.3
    assert x < 8.0
    assert R > 0
    assert B > 0

    if x < 1.1:
        a = 0.574*x**1.61
        b = -0.527*x**1.61
    elif x < 3.3:
        y = x - 1.82
        a = 1 + 0.177*y - 0.504*y**2 - 0.0243*y**3 + 0.721*y**4 + 0.0198*y**5 - 0.775*y**6 + 0.330*y**7
        b = 1.413*y + 2.283*y**2 + 1.072*y**3 - 5.384*y**4 - 0.622*y**5 + 5.303*y**6 - 2.090*y**7
    elif x < 5.9:
        fu = (3.3 / x)**6 * (-0.0370 + 0.0469*B - 0.601*B/R + 0.542/R)
        a = 1.752 - 0.316*x - 0.104*B/((x - 4.67)**2 + 0.341) + fu
        b = -3.09 + 1.825*x + 1.206*B/((x - 4.62)**2 + 0.263)
    elif x < 8.0:
        fu = -0.0447*(x - 5.9)**2 - 0.00978*(x - 5.9)**3
        fh = 0.213*(x - 5.9)**2 + 0.121*(x - 5.9)**3
        a = 1.752 - 0.316*x - 0.104*B/((x - 4.67)**2 + 0.341) + fu
        b = -3.09 + 1.825*x + 1.206*B/((x - 4.62)**2 + 0.263) + fh
    else:
        raise IOError("check your input and make sure its sane, please")
    Alambda = a + b/R
    return Alambda
