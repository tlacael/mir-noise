'''
Functions that return low-level spectral features.

* SpectralCentroid
* SpectralSpread
* SpectralFlatness
* Cepstrum
* MFCC
'''

import numpy as np
from .. import mir_utils
from .. import windowmanager as wm

def SpectralCentroid(x, fftParams):
    ''' x is a time-domain signal. Takes the FFT and returns the vector of spectral
    centroids, based on the fftParams given. '''
    pass

def Centroid(X):
    ''' Returns the centroid/center of mass of X '''
    pass

def SpectralSpread(x, fftParams):
    ''' x is a time-domain signal. Takes the FFT and returns the vector of spectral
    spreads, based on the fftParams given. '''
    pass

def Spread(X):
    ''' Returns the spread of X '''
    pass

def SpectralFlatness(x, fftParams):
    ''' x is a time-domain signal. Takes the FFT and returns the spectral flatness,
    based on the fftParams given. '''
    pass

def Flatness(X):
    ''' Returns the flatness of X '''
    pass

def Cepstrum(X, cepstrumParams):
    ''' Take the Cepstrum of X, return the matrix of Cepstrum coeficients. '''
    pass

def MFCC(X, nFilters, dctWindowSize, minFreq, maxFreq, fftParams):
    ''' Take the MFCC of X, return the matrix of Cepstrum coeficients. 
    X is the 2d spectrogram (FFT frames)
    '''

    nFrames = X.shape[0]
    nFFT = X.shape[1]

    # Get the Mel Filterbank
    filters = wm.MelFilterBank(nFFT, nFilters, minFreq, maxFreq, fftParams)

    # Multiply the Mel Filterbank by the spectrogram to get the Mel Spectrogram
    melSpect = X.dot(filters.transpose())

    # Get the DCT Matrix
    dctMatrix = mir_utils.GetDCTMatrix(nFilters, dctWindowSize)

    # Multilpy the Mel Spectrogram by the DCT Matrix to get the MFCC
    mfcc = melSpect.dot(dctMatrix.T)
    
    return mfcc

def MFCC_Normalized(X, nFilters, dctWindowSize, minFreq, maxFreq, fftParams, nIndexSkip=0):
    
    mfcc = MFCC(X, nFilters, dctWindowSize, minFreq, maxFreq, fftParams, nIndexSkip)
    
    # Remove the end of the mfcc to clear up the data before normalizing
    mfcc_norm = mir_utils.Normalize(mfcc[:, nIndexSkip:])

    return mfcc_norm
