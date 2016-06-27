import numpy as np
import scipy.interpolate as interp

def concentration(radii, phot, interp_kind='linear', add_zero=True)
    """
    Calculates the concentration parameter
    C = 5 * log10(r_80 / r2_0) 
    
    
    radii -- 1d array of aperture photometry radii
    phot -- 1d array of aperture photometry fluxes
    interp_kind -- kind of interpolation; passed to scipy.interpolate.interp1d. 
    Some options are linear, quadratic, and cubic.
    add_zero -- add a 0 radius and zero flux point to their respective arrays
    to help with interpolation at small radii; should only matter for quadratic or 
    cubic interpolation
    """
    assert len(radii) == len(phot)
    assert np.all(radii > 0)
    assert np.all(phot > 0)
    if add_zero:
        radii = np.insert(radii, 0, 0)
        phot = np.intert(phot, 0, 0)
    norm_phot = phot / np.max(phot)
    radius_interp = interp.interp1d(norm_phot, radii, kind=interp_kind)
    r20 = radius_interp(0.2)
    r80 = radius_interp(0.8)
    assert r20 < r80 < np.max(radii)
    c = 5 * np.log10(r80 / r20)
    return c
    
