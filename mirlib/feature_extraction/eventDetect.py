''' @author: Tlacael Esparza'''
''' script analyzes wav audio and detects events 
with novelty function '''


from numpy import *
from matplotlib.pyplot import *
import marlib.matlab as M
import scipy as sp
import math



class onsetDetect:
    
    def __init__(self, x, fs, winLen, hopSize):
        self.x = x
        self.fs = fs
        self.winLen = winLen
        self.hopSize = hopSize
        
    def envelopeFollow(self):
        xBuf = M.shingle(self.x, self.winLen, self.hopSize)
        xBuf.shape = (size(xBuf,0), size(xBuf,1))
        featureLen = size(xBuf, 0)
        
        
        #creat window
        win = hanning(self.winLen)
        self.winMat = tile(win, (featureLen, 1))
        
        #window buffers
        self.xBuf = xBuf * win
        
        #square and mean
        self.xLocEnrg = mean(square(self.xBuf),1)
        
        return self.xLocEnrg
    
    def logDeriv(self):

        xLogDeriv = zeros(size(self.xLocEnrg))
        
        xLogDeriv[0:-1] = log(self.xLocEnrg[1:]) - log(self.xLocEnrg[0:-1])
        xLogDeriv[-1] = 0
        
        xLogDeriv[abs(self.xLocEnrg) < mean(abs(self.xLocEnrg))] = 0
         
        self.xLogDeriv = xLogDeriv
        
        return self.xLogDeriv
    
    def pickPeaks(self):
        
        self.peakMatrix = self.xLogDeriv/max(self.xLogDeriv)
        self.peakMatrix.shape = (1, size(self.peakMatrix))
        
        #set up padding for shifting register
        halfLen = 2
        self.zeroPad = zeros([1,halfLen])
        self.peakWin = size(self.zeroPad,1)*2+1;
        
        self.buff = concatenate((self.zeroPad,self.peakMatrix,self.zeroPad),1)
        self.maxMatrix = zeros([self.peakWin,size(self.peakMatrix,1)])
        
        for i in range(self.peakWin):
            if i == self.peakWin - 1:
                self.maxMatrix[i,:] = self.buff[0,i:]
            else:
                self.maxMatrix[i,:] = self.buff[0,i:1-self.peakWin+i] 
            
        self.maxMatrix = self.maxMatrix.max(0)
        
        compare = z.maxMatrix == z.peakMatrix
        
        peakTimes = #need to finish
        
        
        
            
        
        
        

# load in audio to be analyzed
[x, fs] = M.wavread('/Users/Tlacael/Python/Homework1/mir-noise/feat_extract/RZABR40.wav')
#[x, fs] = M.wavread('GV_Excerp_Short.wav')

winLen = 1024
hopSize = winLen * 0.5

z = onsetDetect(x, fs, winLen, hopSize)

y = z.envelopeFollow()
w = z.logDeriv()
q = z.pickPeaks()
