'''
@author: Christopher Jacoby

misc functions that aren't anywhere else for generic mir things.
'''

from scipy import *
import numpy as np
import marlib.matlab as M
import FFTParams

INT16_MAX = 32768.0

MEL_CONST = 1127.01028
MEL_FREQFACT = 700.

def FreqToMel(freq):
    return MEL_CONST * log(1 + (freq / MEL_FREQFACT))

def MelToFreq(mel):
    return MEL_FREQFACT * (np.power(e, (mel / MEL_CONST)) - 1)

# From http://gitorious.org/loudia/loudia/
def princarg(phase):
    add(phase, pi, phase)
    remainder(phase,-2.0*pi, phase)
    add(phase, pi, phase)
    return phase

def GenTimeVect(fs, N, h):
    ''' Given the sampling rate, window size, and hop size,
    return a vector of the millisecond locatoin of each window. '''
    time = arange(0, N, h)
    time = time / fs
    return time * 1000 # scale to milliseconds

def GetTimesFromHop(fs, h, data):
    timeVect = (arange(len(data)) * h) / float(fs)
    ons = timeVect * array(data, dtype=float32)
    return [x for x in onps if x > 0.0]

def ReadWavFile(filename):
    ''' Read wav data from the file at filename. '''
    return M.wavread(filename)
	
def HalfWaveRectify(x):
	''' Halfwave rectify x:
	    H(x) = (x + |x|) / 2 '''
	return (x + np.absolute(f)) / 2

def Normalize(x):
    ''' Returns x - min(x) / max(x - min(x)), operating in the last dimension '''
    ax = np.asarray(x, dtype=float)
    opAxis = len(ax.shape) - 1
    x_m = (ax.T - ax.min(axis=opAxis)).T
    return (x_m.T / (x_m).max(axis=opAxis)).T

def PrintDataStats(data, title=''):
    dim = data.ndim
    lastaxis = dim - 1
    
    ''' Print statistics for an data source. '''
    dataMax = np.max(data, axis=lastaxis)
    dataMin = np.min(data, axis=lastaxis)
    dataMean = np.mean(data, axis=lastaxis)
    dataStd = np.std(data, axis=lastaxis)
    
    print "Data:", title, " with shape:", data.shape
    print "Range:", dataMax - dataMin, "Min:", dataMin, "Max:", dataMax
    print "Average:", dataMean
    print "StDev:", dataStd
    print

def GenerateTestTone(dur=1.0, freq=440, fs=44100):
    a = 0.95 # amplitude
    t = linspace(0, dur, dur * fs )
 
    return a * sin( freq * t )

def GenerateTestSweep(dur=1.0, startFreq=50, endFreq=1000, fs=44100):
    a = 0.95 # max amplitude
    pow1 = floor(log10(startFreq))
    pow2 = floor(log10(endFreq))
    mult = startFreq / pow(10, pow1)
    
    freqs = mult * logspace(pow1, pow2, dur * fs)

    return a * sin ( cumsum ( 2 * pi * freqs / fs ) )

def AsymmetricTriangularWindow(nLeft, nRight):
    ''' Returns an asymmetric window, with nLeft points to the left of the peak,
    and nRight points to the right of the peak.'''
    left = arange(nLeft + 1, dtype=float)
    if (left > 0).any():
        left /= left.max()
    
    right = arange(nRight + 1, dtype=float)[::-1]
    if (right > 0).any():
        right /= right.max()
        
    # we have to do size + 1 and cut off the last one so that when we normalize it, we loose the one's
    return np.concatenate([left[:-1], np.ones(1), right[1:]])

def MatrixDFT(x, fftParams):
    ''' For given input vector x with shape (N,):
    compute the STFT as a single matrix operation. Returns
    matrix of (time, freq).'''

    N = fftParams.N
    hop = fftParams.h
    zeropad = fftParams.zeropad
    winfunc = fftParams.winfunc
    M = np.int(x.shape[0] / hop)  # M is the number of Hops
    NZ = N + zeropad              # NZ is the size of the buffer after zeropadding
    xmat = np.zeros([M,NZ])       # xmat is the buffered matrix

    # "Buffer" the signal
    for m in range(M):
        x_m = x[m * hop : (m * hop + N)]
        xmat[m, : len(x_m)] = x_m.transpose()
        
    # Create the basis function
    w_k = 2 * np.pi * 1j * np.arange(NZ / 2 + 1)[np.newaxis, :] / float(NZ)
    n = np.arange(NZ)[:, np.newaxis]
    basis = np.exp(w_k * n)
    # Create the zero-padded window 
    w_n = np.concatenate([winfunc(N), np.zeros(zeropad)])[np.newaxis, :]

    # Multiply the window by the buffer, and dot product that with the basis
    # to give you the resultant vector
    return np.dot(xmat * w_n, basis)
    
def GetDCTMatrix(nFilters, dctWindowSize):

    N = np.float(nFilters)
    n = arange(nFilters)
    k = arange(dctWindowSize)

    dctMatrix = np.ones([dctWindowSize, nFilters])

    for i in k:
        dctMatrix[i] = np.cos( (pi * k[i] / N) * (n - .5) )

    return dctMatrix * np.sqrt( 2 / N )

def AverageFeaturesInTime(x, fs, segLength):
    ''' x is an arbitrary signal, or collection of feature vectors, with the first axis
    corresponding to time, and the second to the data. segLength and fs should be in seconds.
    Returns a new vector with time-averaged data, where the resultant segments are of the length specified.
    For instance, if segLength is one, the data will be averaged into a vector of one-second long segments. '''
    nSamps = x.shape[0]
    nFeatures = x.shape[1]
    period = 1 / np.float(fs)
    totalLength = nSamps * period
    sampsInSeg = segLength * fs
    nSegments = np.int(np.ceil(totalLength / np.float(segLength)))

    resultVector = zeros([nSegments, nFeatures])
    old_index = 0

    for i in range(nSegments):
        if old_index + sampsInSeg < nSamps:
            resultVector[i,:] = x[old_index : old_index + sampsInSeg].mean(axis=0)
        else:
            resultVector[i] = x[old_index :].mean(axis=0)

        old_index += sampsInSeg

    return resultVector

