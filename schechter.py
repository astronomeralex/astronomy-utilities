#this will define a schechter function

from numpy import exp

def schechter(l,lstar,alpha):
    """
    this will calculate a value for the Schechter function at a given luminosity (l), and a given lstar and alpha
    """
    x = l/lstar
    return x**alpha * exp(-x)
