'''
   @author: Christopher Jacoby

   PeriodicityDetector.py - this module holds the class for handling 
   low-frequency periodicity detection (beats)
   methods for running each onset detector type. 

   This class also handles peak-picking, etc.
'''

import numpy as np
import windowmanager as wm
from scipy import *
import mir_utils

class BeatDetector:
    def __init__(self, data, dataSamplingRate, noveltyN, noveltyHopDen, noveltyZeroPad,
                 pdpN, pdpHopDen, pdpZeroPad, winfunc=np.ones, debug=False):

        self.data = data
        self.data_fs = dataSamplingRate

        self.novelty_N = noveltyN
        self.novelty_h = noveltyN / np.int(noveltyHopDen)
        self.novelty_ZeroPad = noveltyZeroPad

        self.pdp_N = pdpN
        self.pdp_h = pdpN / np.int(pdpHopDen)
        self.pdp_ZeroPad = pdpZeroPad

        self.winfunc = winfunc
        self.debug=debug
    
    def CalculateNovelty_Grosche(self, C=1000):
        X = mir_utils.MatrixDFT(self.data, self.novelty_N, self.novelty_h, self.novelty_ZeroPad, self.winfunc)
        # calculate Y, the "compressed spectrum"
        Y = log(1 + C * np.absolute(X))
        
        # Calculate Novelty Function
        #  Calculate the derivative of each adjacent Y: ( Y(t+1) - Y(t) )
        #   *** In the actual implementation of this, the authors smoothed it
        dtY = np.diff(Y, axis=0)
        #  Get rid of things < 0
        dtY = dtY * (dtY >= 0)
        #  Get Sum of each vector
        delta = sum(dtY, axis=1)
        # Subtract the mean and keep only the positive.
        noveltyFn = delta - mean(delta)
        noveltyFn = noveltyFn * (noveltyFn >= 0)

        # set the frame rate for later use
        # frame rate is how many novelty-samples there are / second.
        self.frame_fs = self.data_fs / self.novelty_h
        return noveltyFn

    def GetCustomSTFT(self, x, freqs):
        ''' Calculate the STFT for a specific range of freqs '''
        
        xmat, self.pdp_M = wm.BufferSignal(x, self.pdp_N, self.pdp_h, self.pdp_ZeroPad)
        
        # create the basis function: e^-2pijwn
        w_k = (2 * np.pi * 1j * freqs)[np.newaxis, :]
        n = np.arange(self.pdp_N)[:, np.newaxis]
        basis = np.exp(w_k * n)
        
        # windowing
        w_n = self.winfunc(self.pdp_N)[np.newaxis, :]

        self.tempo_fs = self.frame_fs / self.pdp_h
        return np.dot(xmat * w_n, basis)

    def GetFreqPhase_Grosche(self, stft, freqs):
        freq_idx = argmax(np.absolute(stft), axis=1)
        # Calculate omega_t, the frequency that maximizes F
        omega_t = freqs[freq_idx]
        
        F_w_t = stft[np.arange(len(freq_idx)), freq_idx]
        # calculate phase_t, the phase corresponding to that frequency
        phase_t = (1 / (2*np.pi)) * np.arccos( (np.real(F_w_t)) / (np.absolute(F_w_t)))
        
        return omega_t, phase_t

    def GetOptimalPeriodicityKernelsFromFreqPhaseVectors(self, omega_t, phase_t):
        n = np.arange(self.pdp_N)
        K_t = zeros( [len(omega_t), self.pdp_N] )
        for i in range(len(omega_t)):
            K_t[i] = cos( 2 * np.pi * (omega_t[i] * n - phase_t[i]) )
            
            w_n = self.winfunc(self.pdp_N)[np.newaxis, :]    
            
        return K_t * w_n

    def ConstructPDPCurveFromKernels(self, K_t):
        # to make the next part easier to read.
        M = self.pdp_M
        N = self.pdp_N
        h = self.pdp_h
        # include the first half window and end half window?
        size = M * h + N 
        PDP = zeros(size)
        for m in range(M):
            kern = (K_t[m]) * (K_t[m]>0)
            PDP[m * h : (m*h + N)] = kern

        return PDP

    def CalculatePDPCurve(self, noveltyFn):
        N = self.pdp_N
        h = self.pdp_h
        zeroad = self.pdp_ZeroPad
        
        # Get the Freqs to work on, linearly spaced from [30:600] / (60), 
        # so we can run the stft on them.
        freqs = np.linspace(30, 600, N) / (60 * self.frame_fs)
        
        # Calculate the STFT
        tempogram = self.GetCustomSTFT(noveltyFn, freqs)
        
        # Get the freq and phase
        omega_t, phase_t = self.GetFreqPhase_Grosche(tempogram, freqs)
        
        # Calculate optimal periodicity kernels for each
        K_t = self.GetOptimalPeriodicityKernelsFromFreqPhaseVectors(omega_t, phase_t)
        
        # Construct the PLP from the windows
        PDP = self.ConstructPDPCurveFromKernels(K_t)
        
        return np.absolute(tempogram), PDP, K_t

