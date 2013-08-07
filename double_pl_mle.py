#this module will use the MLE technique from
#Howell (2002) to find the parameters for a broken power law
#http://dx.doi.org/10.1016/S0168-9002(02)00794-5

import numpy as np
from scipy.optimize import fmin
#from scipy.optmize import minimize #use for scipy 0.11

def mle(data,guess=[]):
    """
    this will do the minimzation
    guess is the array of initial parameters
    if blank, the code will set it
    """
    
    if len(guess) == 0:
        guess.append(1.)
        guess.append(2.)
        guess.append(np.median(data))
        guess = np.array(guess)
        
    return fmin(mloglike,guess,[data])

def mloglike(params,data):
    """
    params is an array the contains three variables, alpha1, alpha2 and Ek, the break point
    data is an array that contains the data
    this function returns the -Log likelihood (equation 7 of Howell) so that the minimzation
    of this function returns the maximum likelihood parameter array
    """
    alpha1 = params[0]
    alpha2 = params[1]
    Ek = params[2]
    
    
    firstpart = - float(len(data)) * np.log(A(params,data))
    
    secondpart = alpha1 * np.sum(np.log( data[np.where(data < Ek)] / Ek ))
    
    thirdpart = alpha2 * np.sum(np.log( data[np.where(data >= Ek)] / Ek ))
    
    return firstpart + secondpart + thirdpart
    
def A(params,data):
    """
    this function takes in the params array and data array
    it returns the normalization A, which is equation 4 of Howell
    """
    
    alpha1 = params[0]
    alpha2 = params[1]
    Ek = params[2]
    E1 = np.min(data)
    E2 = np.max(data)
    
    top = (alpha1 - 1.0) * (alpha2 - 1.0)
    bottom = Ek * (alpha1 - alpha2 + (alpha2 - 1)*(E1 / Ek)**(1.0 - alpha1) - (alpha1 - 1.0)*(E2/Ek)**(1.0 - alpha2) )
    
    return top / bottom
    
