#this module will computer various MLEs for distributions

import numpy as np

def pareto(data):
    """
    this will return the MVUE's and variances from Feigelson and Babus pg. 90
    data is a numpy array of the data
    """
    
    bhat = np.min(data)
    alphahat = len(data) / np.sum(np.log(data / bhat))
    
    alphastar = (1. - 2. / len(data))*alphahat
    bstar = bhat*(1. - 1. / (alphahat* (len(data) - 1.)))
    

def 
