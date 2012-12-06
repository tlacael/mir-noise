'''
 @author: Christopher Jacoby

 Routines to handle matlab data
'''

import numpy as np
from scipy import *
from scipy.io import loadmat

MATLAB_DATA_KEY = "T"
        
def GetMatlabVectors(file):
    ''' The .mat files are processed with scipy.io.loadmat.
        This returns a dict, with the keys as variable names.
        The one we care about is called T, so we're just 
          returning the contents of that '''
    matFile = file[:-4] + ".mat"
    matdict = loadmat(matFile)

    data = array(matdict[MATLAB_DATA_KEY])

    # This is because the stupid .mat files are in different formats
    if len(data.shape) > 1:
        data.shape = (max(data.shape),)    

    return data

def ConvertMatlabVectors(samplingRate, vector):
    ''' But in order to draw the matlab vectors nicely, it needs
         to be x = time, with 1s where the onsets are and 0s otherwise. '''
    sampVector = array(vector * samplingRate, dtype=int32)
    x = linspace( 0, max(sampVector), len(sampVector) * 100)
    y = zeros(len(x))
    for n in range(len(sampVector)):
        y[np.searchsorted(x, sampVector[n])] = 1

    return array([x, y])

