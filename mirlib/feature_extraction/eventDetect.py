''' @author: Tlacael Esparza'''
''' script analyzes wav audio and detects events 
with novelty function, returns index with beginning and ending time markings 
for identified event'''


from numpy import *
from matplotlib.pyplot import *
import marlib.matlab as M
import scipy as sp
import math



class onsetDetect:
    
    def __init__(self, x, fs):
        self.x = x
        self.fs = fs

    def envelopeFollow(self, winLen, hopSize):
        self.winLen = winLen
        self.hopSize = hopSize
        
        xPad = zeros((self.winLen/2,1))
        xPad = concatenate((xPad, self.x),0)
            
        
        xBuf = M.shingle(xPad, self.winLen, self.hopSize)
        xBuf.shape = (size(xBuf,0), size(xBuf,1))
        featureLen = size(xBuf, 0)
        
        
        #creat window
        win = hanning(self.winLen)
        self.winMat = tile(win, (featureLen, 1))
        
        #window buffers
        self.xBuf = xBuf * win
        
        #square and mean
        self.xEnvelope = mean(abs(self.xBuf),1)
        
        return self.xEnvelope
    
        
    def envelopeFollowEnergy(self, winLen, hopSize):
        self.winLen = winLen
        self.hopSize = hopSize
        
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
    
    def pickPeaks(self, peakMatrix):
        
        self.peakMatrix = peakMatrix/max(peakMatrix)
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
            
        self.maxMatrix[self.maxMatrix==0]=-1
        
        self.maxMatrix = self.maxMatrix.max(0)
        
        self.compare = z.maxMatrix == z.peakMatrix
        
        self.peakTimes = nonzero
        
    def findLenofEvent(self):
        #set 2 second window for strong smoothing
        winLen = self.fs * 2
        hopSize = winLen
        
        EnvSmooth = self.envelopeFollowEnergy(winLen, hopSize)
        #normalize
        EnvSmooth = divide(EnvSmooth, EnvSmooth.max()) 
        thresh = mean(EnvSmooth)
        
        EnvSmooth[EnvSmooth<thresh]=0
        
        
        EventCenters = np.nonzero(EnvSmooth)
        EventCenters = np.array((EventCenters))
        
        self.cents = EventCenters
        
        
        self.numberOfEvents = EventCenters.size
        
        EventTimes = zeros((self.numberOfEvents,2))
        eventIndex = arange(self.numberOfEvents)
        
        #convert event centers to time windows in seconds
        
        widen = 1 #amount to pad window on either side of event, in seconds
        for i in eventIndex:
            EventTimes[i,0] = (EventCenters[0,i] - 0.5)*winLen/float(self.fs)-widen
            EventTimes[i,1] = (EventCenters[0,i] + 0.5)*winLen/float(self.fs)+widen
            
        self.EventTimes = EventTimes
        
        reducedEvents = zeros((self.numberOfEvents,2))
        #reduce number of segments i.e. overlaps, nearby
        temp = EventTimes
        
        offset =0;
        for i in eventIndex:
            
            reducedEvents[i-offset,0] = temp[i,0]
            
            if i < eventIndex[-1]:
                if (temp[i+1,0] - temp[i,1]) < 2:
                    reducedEvents[i-offset,1] = temp[i+1,1]
                    temp[i+1,:] = reducedEvents[i-offset,:]
                    offset +=1
                    
                    
                else:
                    reducedEvents[i-offset,1] = temp[i,1]
              
        self.reducedEvents = reducedEvents[:-offset,:]
        self.numberOfEvents = size(self.reducedEvents,0)  
            
        return self.reducedEvents
                
        
        
        
''' Test script
# load in audio to be analyzed
#[x, fs] = M.wavread('/Users/Tlacael/Python/Homework1/mir-noise/feat_extract/RZABR40.wav')
#[x, fs] = M.wavread('GV_Excerp_Short.wav')
[x, fs] = M.wavread('wburgShort.wav')



winLen = fs*2
hopSize = winLen 

z = onsetDetect(x, fs)
#y = z.envelopeFollow(winLen, hopSize)
y = z.envelopeFollowEnergy(winLen, hopSize)
timeX = arange(z.x.size)
timeX.shape = z.x.shape
timeX[::40] = divide(timeX[::40], fs)
timeX[::40].shape = z.x[::40].shape
plot(timeX[::40], z.x[::40], color='r')

y= divide(y,y.max())
timeY = range(y.size)
timeY = divide(timeY, (fs/hopSize))
timeY.shape = y.shape
plot(timeY,y, color='b');show()

#w = z.logDeriv()
q = z.pickPeaks(y)
'''
