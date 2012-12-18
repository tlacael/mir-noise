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

def SpectralFlux(X, fftParams, bRectified=True):
    ''' Spectral Flux SF(m) = (2/N) sum[0:N/2] H(|Xk(m)| - |Xk(m-1)|)
    H(x) = (x + |x|) / 2'''
    
    N = fftParams.N
    magX = abs(X)
    diffX = np.diff(magX, axis=0)

    if bRectified:
        diffX = mir_utils.HalfWaveRectify(diffX)

    return (2 / np.float(N)) * np.sum(diffX[:, :(N/2)], axis=1)

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

def MFCC(X, mfccParams, fftParams):
    ''' Take the MFCC of X, return the matrix of Cepstrum coeficients. 
    X is the 2d spectrogram (FFT frames)

    taken from Juan's MIR notes.
    '''
    X = np.array(X)  
    nFrames = X.shape[0]
    nFFT = X.shape[1]
     
     # take magnitude and put into DB
    XmagDB = 20*np.log10(abs(X)+0.0000001)

    # Get the Mel Filterbank
    filters = wm.MelFilterBank(nFFT, mfccParams.nFilters, mfccParams.minFreq, mfccParams.maxFreq, fftParams)

    # Multiply the Mel Filterbank by the spectrogram to get the Mel Spectrogram
    melSpect = XmagDB.dot(filters.transpose())

    # Get the DCT Matrix
    dctMatrix = mir_utils.GetDCTMatrix(mfccParams.nFilters, mfccParams.nDCTCoefs)

    # Multilpy the Mel Spectrogram by the DCT Matrix to get the MFCC
    mfcc = melSpect.dot(dctMatrix.T)
    
    return mfcc[:,2:]

def MFCC_Normalized(X, mfccParams, fftParams):
    
    mfcc = MFCC(X, mfccParams, fftParams)
    
    
    # Remove the end of the mfcc to clear up the data before normalizing
    mfcc_norm = mir_utils.Normalize2(mfcc)

    return mfcc_norm
