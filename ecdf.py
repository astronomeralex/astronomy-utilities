#this module is helpful in plotting the ecdf of a 1-D dataset

import numpy as np

def ecdf(arr):
    """
    this will take in a numpy array arr
    and return x, y, and the variance, which can then be used to
    plot the ecdf
    """
    newarr = np.sort(arr)
    xout = []
    yout = []
    
    stepsize = 1.0 / len(newarr)
    
    for i,value in enumerate(newarr):
        xout.append(value)
        yout.append(i * stepsize)
        xout.append(value)
        yout.append((i+1)*stepsize)
    xout = np.array(xout)
    yout = np.array(yout)
    variance = yout * (1-yout) / len(yout)
    return xout,yout,variance
