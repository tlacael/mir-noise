'''
   @author: Christopher Jacoby

   peakdetector.py: handles peak detection of input data
'''

import numpy as np
from windowmanager import *

def GetPeakTimes(data, fs, delta_thresh=.75, alpha_thresh=.75, w=3, m=3):
    ''' Returns the times of the peaks, given the data and the sampling rate '''
    pp = PeakPicking()

    d1 = pp.PostProcessing(data)
    peaks = pp.PeakPicking(d1, delta_thresh, alpha_thresh, w, m)
    peaks = array(peaks.nonzero()[0], dtype=float32) / fs
    
    return peaks

def GetPeakTimes_Adaptive(data, fs):
    ''' Return the times of the peaks, calculated using an adaptive algorithm. '''

    return

def CalculateOnsets(self, logEnergy, rectspectflux, rectcomplex, delta=.75):

    pp = PeakPicking()

    # PostProcessing
    d1 = pp.PostProcessing(logEnergy)
    d2 = pp.PostProcessing(rectspectflux)
    d3 = pp.PostProcessing(rectcomplex)
        
    # Thresholding

    # Peak-picking
    d1 = pp.PeakPicking(d1, delta, .75, 3, 3)
    d2 = pp.PeakPicking(d2, delta, .75, 3, 3)
    d3 = pp.PeakPicking(d3, delta, .75, 3, 3)
    
    # Select peaks for each set
    return (d1, d2, d3)

class PeakPicking:
    def __init__(self):
        pass

    def PostProcessing(self, data):
        return self.Normalize(data)

    def Normalize(self, data):
        ''' Set the mean of the dataset to 0, and the standard dev to 1'''

        tmpData = (data - mean(data)) / std(data)
        return tmpData

    def Thresholding(self, data):
        pass
        

    def PeakPicking(self, data, delta, alpha, w=3, m=3, debug=False):
        ''' Returns an array of 1's and 0's for each window, specifying whether or not the value is a peak or not.

        Required conditions:
        (see respective functions), PeakPicking_{1,2,3} each return an array of (int) truth values, one for each window.
        1, 2, 3 must all be true at each location to return a peak

        returns: an array of (int) 1's and 0's with 1's representing peaks and 0's representing... well, not.'''

        result = []

        windowMgr = WindowManager(data, (w*2 + 1), 1, None)
        while True:

            # First Condition
            first = self.PeakPicking_1(windowMgr, w)

            # Second Condition
            second = self.PeakPicking_2(windowMgr, delta, w, m)
            
            # Third Condition
            third = self.PeakPicking_3(windowMgr, alpha)

            result.append( first and second and third )

            if not windowMgr.Hop():
                break

        return array(result)

        ''' Peak Picking Function variables:
         w = 3, the size of the window for the local maximum
        m = 3, a  multiplier so that the mean is calculated over a larger range before the peak
        delta is the local threhshold
        g_alpha(n) is a threhsold function
        delta and alpha are just parameters to set'''

    def PeakPicking_1(self, windowMgr, w):
        ''' f(n) >= f(k) for all k such that n - w <= k <= n + w '''

        window = windowMgr.GetArbitraryWindow(w, w)

        f_k = window[len(window) / 2]
        return (f_k >= window[:w]).all() and (f_k >= window[w+1:]).all()

    def PeakPicking_2(self, windowMgr, delta, w, m):
        ''' f(n) >= (sum[k=n - mw, n + w] f(k) / (mw + w + 1)) + delta '''
        
        window = windowMgr.GetArbitraryWindow(w * m, w)

        f_n = windowMgr.GetSampleAtWindow()
        f_k = ( sum(window) / (m * w + w + 1)) + delta

        return (f_n >= f_k)

    def PeakPicking_3(self, windowMgr, alpha):
        ''' f(n) >= g_alpha (n-1)
        g_alpha(n) = max(f(n), alpha*g_alpha(n - 1) + (1 - alpha)f(n)) 

        Okay, after staring at this for a while, I can't figure out how to implement it.
        f(n) is a function of g_alpha(n-1), but g_alpha(n) is a function of g_alpha(n-1),
        making it recursive, which I guess could be okay, but for now, I'm just going to
        ignore this condition.'''

        #f_n_min1 = windowMgr.GetSampleAtWindow(-1)
        #g_alpha_n_minus_1 = max()
        return True

    def MovingSquare(self, windowMgr, delta, const_lambda, M):
        ''' A form of adaptive thresholding using the square of the window.
        delta_hat(n) = delta + lambda * square (window) '''

        window = windowMgr.GetArbitraryWindow(M, M)
        window = pow(a, 2) * np.hanning(len(window))

        f_n = windowMgr.GetSampleAtWindow()
        f_k = delta + const_lambda * window

        return (f_n >= f_k)

    def MovingMedian(self, windowMgr, delta, const_lambda, M):
        ''' A form of adaptive thresholding, using the median of the window.
        delta_hat(n) = delta + lambda * median( window )'''

        window = windowMgr.GetArbitraryWindow(M, M)

        f_n = windowMgr.GetSampleAtWindow()
        f_k = delta + const_lambda * np.median( window )

        return (f_n >= f_k)
        
