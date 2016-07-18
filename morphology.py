import numpy as np
import scipy.interpolate as interp

def concentration(radii, phot, eta_radius=0.2, eta_radius_factor=1.5, interp_kind='linear', add_zero=True):
    """
    Calculates the concentration parameter
    C = 5 * log10(r_80 / r2_0) 
    Inputs:
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
        phot = np.insert(phot, 0, 0)
    eta_interp = interp.interp1d(eta(radii, phot), radii, kind=interp_kind)
    eta_r = eta_radius_factor * eta_interp(eta_radius)
    phot_interp = interp.interp1d(radii, phot, kind=interp_kind)
    maxphot = phot_interp(eta_r)
    norm_phot = phot / maxphot
    radius_interp = interp.interp1d(norm_phot, radii, kind=interp_kind)
    r20 = radius_interp(0.2)
    r80 = radius_interp(0.8)
    assert r20 < r80 < np.max(radii)
    c = 5 * np.log10(r80 / r20)
    return c
    
def eta(radii, phot):
    """
    eta = I(r) / \bar{I}(<r)
    
    radii -- 1d array of aperture photometry radii
    phot -- 1d array of aperture photometry fluxes
    
    this is currently calculated quite naively, and probably could be done better
    """
    phot_area = np.pi * radii**2
    phot_area_diff = np.ediff1d(phot_area, to_begin=phot_area[0])
    I_bar = phot / (phot_area)
    I_delta_r = np.ediff1d(phot, to_begin=phot[0]) / phot_area_diff
    I_r = (I_delta_r[:-1] + I_delta_r[1:]) / 2 #lost last array element here
    I_r = np.append(I_r, I_delta_r[-1]) #added it back in here
    eta = I_r / I_bar
    return eta

