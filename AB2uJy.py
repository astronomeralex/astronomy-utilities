#this module will convert from AB magnitudes to microJy
#written by Alex Hagen
# hagen@psu.edu

# licensed under Attribution-NonCommercial 4.0 International
#https://creativecommons.org/licenses/by-nc/4.0/legalcode

# wikiepdia reference: https://en.wikipedia.org/wiki/Jansky
#fnu (uJy) = 10**((23.9 - AB) / 2.5)

from numpy import abs, array

def AB2uJy(abmag,abmagerr):
    """
    converts an AB magnitude to flux density in microJy
    abmag, and abmagerr are both floats or arrays of floats
    """
    fnu = 10**((23.9 - abmag) / 2.5)
    fnu_upper = 10**((23.9 - abmag + abmagerr) / 2.5)
    fnu_lower = 10**((23.9 - abmag - abmagerr) / 2.5)
    fnuerr = (abs(fnu - fnu_upper) + abs(fnu - fnu_lower)) / 2.0
    return array([fnu, fnuerr])
    
    
