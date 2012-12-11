'''
   @author: Christopher Jacoby

   OnsetDetector.py - this module holds the primary workhorse for Onset Detecting, including 
   methods for running each onset detector type. 

   This class also handles peak-picking, etc.
'''

import scipy.io.wavfile as wvio
import numpy as np
from windowmanager import WindowManager
from scipy import *
from mir_utils import *

INT16_MAX = 32768.0

class OnsetDetector:
    def __init__(self, filename, N=1024, h=512, window=np.hanning):
        self.ReadWav(filename)
        self.N = N
        self.h = h
        self.window = window

    def ReadWav(self, filename):
        dataread = wvio.read(filename)
        self.fs = dataread[0]
        self.data = np.divide(dataread[1], INT16_MAX)

    def Energy(self, rawdata):
        ''' Input: Raw wav data
            returns: localEnergy, E(m) '''

        windowMgr = WindowManager(rawdata, self.N, self.h)
        wind = windowMgr.NextWindow()
        result = []
        while ( wind is not None):
            result.append(np.power(wind, 2.0).mean() )
            wind = windowMgr.NextWindow()

        return array(result)

    def EnergyDerivative(self, localEnergy):
        ''' Input: Local Energy
            returns: local energy derivative, E(m)' '''
        return np.diff(localEnergy)

    def LogEnergyDerivative(self):
        ''' Log Energy Derivative:
            dE(m)/dm
            --------  = dlog(E(m)) / dm
               E(m)
            
            E(m) = (1/N) sum[-N/2->N/2] (x(n-mh))^2 * w(n)
               '''

        # Calculate the Energy, E(m)
        localEnergy = self.Energy(self.data)
        
        # Calculate the log Energy Derivative (... take the log)
        logEnergy = log(localEnergy)

        # Calculate the dEnergy, for printing purposes
        dEnergy = self.EnergyDerivative(localEnergy)
        
        # Calculate the Energy derivative, dlogE(m)/dm
        dLogEnergy = self.EnergyDerivative(logEnergy)
        # clear inf's to make further things cleaner
        dLogEnergy = array([ (0 if x==inf else x) for x in dLogEnergy])
                
        return dLogEnergy, dEnergy, localEnergy

    def HalfWaveRect(self, x):
        ''' Halfwave rectify x:
            H(x) = (x + |x|) / 2 '''
        return (x + np.absolute(x)) / 2

    def DoForAllWindows(self, procFunc, freqMode=True):
        result = []

        windowMgr = WindowManager(self.data, self.N, self.h)

        if freqMode:
            thisWind = windowMgr.NextFFTWindow()
        else:
            thisWind = windowMgr.NextWindow()

        while ( thisWind is not None ):
            result.append( procFunc(thisWind) )

            if freqMode:
                thisWind = windowMgr.NextFFTWindow()
            else:
                thisWind = windowMgr.NextWindow()
        
        return array(result)

    def RectifiedSpectralFlux_Loop(self, thisWind):
        # Get the difference between this window and the last window ( |Xk(m)| - |Xk(m-1)} )

        workingWind = thisWind[:len(thisWind) / 2]
        fftDiff = np.absolute( workingWind ) - np.absolute(self.lastwindow)
            
        # Half-wave Rectify!
        rectRes = self.HalfWaveRect(fftDiff)
        
        # Store this window as the previous window
        self.lastwindow = workingWind
                
        # Do the sum (2/N) sum[0:N/2] H(x)
        return (2. / self.N) * sum(rectRes) 
        
    def RectifiedSpectralFlux(self):
        ''' Rectified Spectral Flux:
            SF(m) = (2/N) sum[0:N/2] H(|Xk(m)| - |Xk(m-1)|)
            H(x) = (x + |x|) / 2 '''

        self.nexttolastwindow = zeros(self.N / 2)
        self.lastwindow = zeros(self.N / 2)
        return self.DoForAllWindows(self.RectifiedSpectralFlux_Loop)

    def RectifiedComplex_Loop(self, thisWind):
        # Get X_k(m) - the complex domain for this window
        # Get X^_k(m) - complex domain thingy for the last window

        X_k = thisWind[:len(thisWind) / 2]

        # Get X^_k(m)
        anglehat_k = princarg(2 * angle(self.lastwindow) - angle(self.nexttolastwindow))
        Xhat_k = absolute(self.lastwindow) * exp( 1j * anglehat_k)

        # Do the adjacent window difference
        complexDiff = np.absolute(X_k - Xhat_k)

        # Rectify
        rectTruth = array(np.greater_equal(absolute(X_k), absolute(self.lastwindow)), dtype=int)
        rectifiedComplexDiff = complexDiff * rectTruth

        # Update the previous windows
        self.nexttolastwindow = self.lastwindow
        self.lastwindow = X_k

        # Rectify, and store function
        return (2. / self.N) * sum(rectifiedComplexDiff)
        

    def RectifiedComplex(self):
        ''' Complex Domain combinses the previous, the Spectral Flux, with phase deviation.

        RCD_k(m):
        if |X_k(m)| >= |X_k(m-1)|:
             |X_k(m) - Xt_k(m)|
        else:
             0
             
        X_k(m) = |X(m)|e^j*phase(m)
        Kt_k(m) = |X(m-1)|e^j*phasediff(m-1)'''
        
        self.lastwindow = zeros(self.N / 2)
        return self.DoForAllWindows(self.RectifiedComplex_Loop)

    def CalculateOnsets(self, logEnergy, rectspectflux, rectcomplex, delta=.75):

        # PostProcessing
        d1 = self.PostProcessing(logEnergy)
        d2 = self.PostProcessing(rectspectflux)
        d3 = self.PostProcessing(rectcomplex)
        
        # Thresholding

        # Peak-picking
        d1 = self.PeakPicking(d1, delta, .75, 3, 3)
        d2 = self.PeakPicking(d2, delta, .75, 3, 3)
        d3 = self.PeakPicking(d3, delta, .75, 3, 3)

        # Select peaks for each set
        return (d1, d2, d3)

    def PostProcessing(self, data):
        return self.Normalize(data)

    def Normalize(self, data):
        ''' Set the mean of the dataset to 0, and the standard dev to 1'''

        tmpData = (data - mean(data)) / std(data)
        return tmpData

    def Thresholding(self, data):
        pass
        

    def PeakPicking(self, data, delta, alpha, w=3, m=3):
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
