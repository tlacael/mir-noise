'''
 @author: Christopher Jacoby

 windowmanager.py contains the WindowManager class, which deals with getting windows of a signal
 for various processing. 
'''
import numpy as np
import FFTParams
from mir_utils import *
from scipy import *

class WindowManager:
    def __init__(self, data, N, h, win=np.hanning):

        self.N = N
        if self.N % 2 != 0: # N is odd:
            self.pointer = -((self.N - 1) / 2)
        else:
            # if log2(self.N) % 1 == 0: # N is a power of 2... or just even
            self.pointer = -(self.N / 2)

        self.data = data
        self.h = h
        self.window = win

    def NextWindow(self):
        ''' I have chosen to make self.pointer start at the 'beginning', which is half of a window before the actual zero. Numbers before the beginning are set to zero.

        At the end, pointer actually has to go to the end, and numbers after the end are also set to zero.

        This function does hop the pointer.'''
        
        # Get the window
        # Case 1: The beginning - put zeros at the front
        if self.pointer < 0:
            ret = zeros(self.N)
            ret[ (self.N/2 - self.pointer) : ] = self.data[ : self.N/2 + self.pointer]

        # Case 2: The End - put zeros at the end
        elif (self.pointer + self.N) >= len(self.data):
            ret = zeros(self.N)
            ret[ : (len(self.data) - self.pointer) ] = self.data[ self.pointer : ]
            
        else: # everything else - normal case
            ret = self.data[self.pointer : self.pointer + self.N]

        # Apply windowing
        if self.window is not None:
            ret = ret * self.window(self.N)
            
        # Hop!
        if self.Hop():
            return ret
        else:
            return None

    def NextFFTWindow(self):
        ''' Returns the next window, FFT'd

        This function does hop the pointer.'''
        wind = self.NextWindow()

        if wind is not None:
            return np.fft.fft(wind)
        else:
            return None

    def GetArbitraryWindow(self, nBeforeK, nAfterK):
        ''' This function returns and asymetrical window
            Returns a window containing [n values before the pointer, the pointer, n values after the pointer].

            So... conceptually, in order to make my normal windowing work correctly, this gets a little complicated.
            In normal windowing, the time k, the time which the window represents is the middle of the window. However, in order to make
            the normal windows easier to calculate, the pointer actually points to the first sample in the window, not the time of the window.

            Therefore, for this function, in order to get an arbitrary window around time k, we need to add N / 2 to the pointer, 
            and then get the window before and after based on that. *sigh* That made calculating GetNextWindow() a lot easier, but this a
            tad more complicated. So it goes.

        If you use this function alone, you must hop manually. '''

        k_ptr = self.pointer + (self.N / 2)
        lft_ptr = k_ptr - nBeforeK
        rgt_ptr = k_ptr + nAfterK

        # Get the left side
        if lft_ptr < 0:
            leftSide = zeros(nBeforeK)
            if k_ptr > 0:
                leftSide[ -k_ptr : ] = self.data[:k_ptr]
        else:
            leftSide = self.data[lft_ptr:k_ptr]

        # Get the right side
        if rgt_ptr > len(self.data):
            rightSide = zeros(nAfterK)
            if k_ptr < len(self.data) - 1:
                rightSide[ : (len(self.data) - k_ptr)] = self.data[k_ptr + 1:]
        else:
            rightSide = self.data[k_ptr + 1 : rgt_ptr + 1]

        # Get the value at k
        k = self.data[k_ptr]
        
        # put 'em all together, and what have you got?      
        result = np.append(leftSide, k)
        result = np.append(result, rightSide)

        return result
    
    def GetSampleAtWindow(self, offset=0):
        
        k = self.pointer + (self.N / 2) + offset
        
        if k < 0 or k > len(self.data):
            return 0.0
        else:
            return self.data[k]

    def Hop(self):
        ''' Increment the window pointer by the hop size.

            returns:
              True if the operation was successful
              False if this would put us past the end of the window'''

        ret = False
        if ( ((self.pointer + self.h) + (self.N/2)) < len(self.data)):
            self.pointer = self.pointer + self.h
            ret = True

        return ret

def BufferSignal(x, N, h, zeropad=0, padEnds=True):
    ''' Buffer a signal a la matlab. '''

    newX = x

    # Prepend and append zeros S.T. the first and last windows are centered at t=0 and t=len
    if padEnds:
        frontPad = zeros(N/2)
        totalX = len(frontPad) + len(x)
        endPad = zeros( ceil(((totalX) / float(N/2)) * (N/2)) - totalX)
        newX = np.concatenate([frontPad, x, endPad])

    # Buffer the signal
    M = np.int(newX.shape[0] / h)  # Number of hops
    NZ = N + zeropad
    xmat = np.zeros([M, NZ])        # buffered matrix

    for m in range(M):
        x_m = newX[m * h : (m * h + N)]
        xmat[m, : len(x_m)] = x_m.transpose()

    return xmat, M

class FilterBank:
    ''' Assume we're given the entire fft window, so only half of it is useful...'''
    
    def __init__(self, fftParams, nFilters, minFreq, maxFreq, freqSelect='linear'):
        self.fs = fftParams.fs
        self.N = fftParams.N
        
        if minFreq < 0 or maxFreq < 0:
            raise ValueError("Invalid Frequency < 0 [%d, %d]" %(minFreq, maxFreq))
        elif minFreq > self.fs or maxFreq > self.fs:
            raise ValueError("Cannot have value greater than the sampling frequency %ld" % (fs))
        elif minFreq >= maxFreq:
            raise ValueError("Cannot have a min freq >= max freq")
        if freqSelect not in ['linear', 'log', 'mfcc']:
            raise ValueError("Invalid type: %s" % (type))
        if nFilters < 1:
            raise ValueError("Must have a positive number of center freqs.")
        elif type(nFilters) is not int:
            raise TypeError("n freqs must be an int")

        self.nFilters = nFilters
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.freqSelect = freqSelect

    def CalculateCenterFreqs(self):
        ''' Returns an sorted array of N+2 center freqs, such that:
          * the first filter begins at the min freq, and ends at the n[0]+1'th filter's center freq
          * The last filter begins at the n[len-1]'s center freq, and ends at max'''
        centerFreqs = array([])

        if self.freqSelect is 'log':
            centerFreqs = np.logspace(log10(self.minFreq), log10(self.maxFreq), self.nFilters + 2)
        elif self.freqSelect is 'mfcc':
            minMel = FreqToMel(self.minFreq)
            maxMel = FreqToMel(self.maxFreq)
            melFreqs = np.linspace(minMel, maxMel, self.nFilters+2)
            centerFreqs = MelToFreq(melFreqs)
        else: # assume type is linear
            centerFreqs = np.linspace(self.minFreq, self.maxFreq, self.nFilters+2)

        return centerFreqs

    def GetFilters(self, fftN):
        ''' self.N is different from fftN, because we need self.N to be the full length of the fft,
        in order to get the accurate frequecies. fftN is the size of N/2 of the fft window, so we're not
        running calculations on the redundant data. '''
        
        self.filterBank = np.zeros([self.nFilters, fftN])

        # First get a list of center frequencies
        centerFreqs = self.CalculateCenterFreqs()

        # Get the bin frequencies
        binFreqs = (self.fs / (self.N)) * arange(fftN)

        for index in range(self.nFilters):
            # Get the start and end FFT index for this filter
            minInd = binFreqs.searchsorted(centerFreqs[index])
            centerInd = binFreqs.searchsorted(centerFreqs[index+1])
            maxInd = binFreqs.searchsorted(centerFreqs[index+2])

            winL = centerInd - minInd
            winR = maxInd - centerInd
            
            wind = AsymmetricTriangularWindow(winL, winR) # in mir_utils
            # Normalize the window to 1
            # technically we need to do an abs first, but I don't care 'cause the window function
            # only gives values from 0-1 anyway
            wind = wind / sum(wind)

            self.filterBank[index, minInd:maxInd+1] = wind
        
        return self.filterBank

def MelFilterBank(N, nFilters, minFreq, maxFreq, fftParams):
    ''' Given an FFT window, return a matrix of filters, based on the input parameters. 
    This function simply returns the matrix to multiply; it doesn't do the multiplication itself.'''

    flt = FilterBank(fftParams, nFilters, minFreq, maxFreq, 'mfcc')
    return flt.GetFilters(N)
