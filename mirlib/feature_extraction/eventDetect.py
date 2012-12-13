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
        
        if x.ndim == 2 and x.shape[1] == 1:
            x.shape = (x.shape[0])            
            
        self.x = x
        self.fs = fs

    def envelopeFollow(self, winLen, hopSize):
        self.winLen = winLen
        self.hopSize = hopSize

    
        print self.x.shape
        print 'hi'
        xPad = zeros((self.winLen/2))
        print xPad.shape
      #  size(xPad,1) = size(self.x,1)
        xPad = concatenate([xPad, self.x])
            
        
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
        

        
        xPad = zeros(self.winLen/2)
        
        print shape(self.x)
        print shape(xPad)
       # size(xPad,1) = size(self.x,1)
        xPad = concatenate([xPad, self.x, xPad])
        
        xBuf = M.shingle(xPad, self.winLen, self.hopSize)
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
        halfLen = 4
        self.zeroPad = zeros([1,halfLen])
        self.peakWin = size(self.zeroPad,1)*2+1;
        
        self.buff = concatenate((self.zeroPad,self.peakMatrix,self.zeroPad),1)
        self.maxMatrix = zeros([self.peakWin,size(self.peakMatrix,1)])
        
        for i in range(self.peakWin):
            if i == self.peakWin - 1:
                self.maxMatrix[i,:] = self.buff[0,i:]
            else:
                self.maxMatrix[i,:] = self.buff[0,i:1-self.peakWin+i] 
            
        self.maxMatrix[self.maxMatrix==0] = -1
        
        self.maxMatrix = self.maxMatrix.max(0)
        
        self.compare = self.maxMatrix == self.peakMatrix
        
        self.peakTimes = nonzero(self.compare)
        self.peakTimes = array((self.peakTimes))
        
        return self.peakTimes[1,:]
        
    def findEventLocations(self):
        #set 2 second window for strong smoothing
        winLen = self.fs*0.5
        hopSize = winLen/2.
        
        self.EnvSmooth = self.envelopeFollowEnergy(winLen, hopSize)
        
        #normalize
        #self.EnvSmooth = divide(EnvSmooth, EnvSmooth.max()) 
        #thresh = mean(EnvSmooth)
        
        
        EnvThresh = self.EnvSmooth
        thresh = median(EnvThresh)*1.4
        
        EnvThresh[EnvThresh<thresh]=0
        
        
        EventCenters = nonzero(EnvThresh)
        EventCenters = array(EventCenters)
    
        lengths = zeros((EventCenters.size,2))
        count = 0
        len=0
        
        '''
        for sample in arange(EnvThresh.size):
            while(EnvThresh[sample]!= 0):
                len+=1
            lengths[count] = (len,sample-len)
            if len >0:
                count+=1
                len=0
        self.lengths = lengths
        
        '''
        
        EventCenters = array((EventCenters))
        
        self.cents = EventCenters
        
        #EventCenters = array(peakTimes)
        self.cents = EventCenters
        
        self.numberOfEvents = EventCenters.size
        
        EventTimes = zeros((self.numberOfEvents,2))
        eventIndex = arange(self.numberOfEvents)
        
        chunkLen = self.fs*self.x.size
        
        #convert event centers to time windows in seconds
        
        widen = 1 #amount to pad window on either side of event, in seconds
        padAdjust = 0.5
        for i in eventIndex:
            EventTimes[i,0] = (EventCenters[0,i] - 0.5 - padAdjust)*hopSize/float(self.fs)-widen
            EventTimes[i,1] = (EventCenters[0,i] + 0.5- padAdjust)*hopSize/float(self.fs)+widen
        
        #make sure time vlues not out of bounds
        EventTimes[EventTimes<0]=0
        
        EventTimes[EventTimes > chunkLen]=chunkLen
        self.EventTimes = EventTimes
        
        
        eventIndex = arange(self.numberOfEvents)
        #reduce number of segments i.e. overlaps, nearby
        reducedEvents = zeros((self.numberOfEvents,2))
        temp = self.EventTimes
        
        offset =0;
        timeThresh = 0#1*winLen/float(self.fs) #in seconds
        i = 1
        
        reducedEvents[0,:] = temp[0,:]
        while((i +offset+1) < self.numberOfEvents):
            if temp[i+offset,0]-reducedEvents[i-1,0] <=0:
                reducedEvents[i-1,1] = temp[i+offset,1]
                offset+=1
            elif temp[i+offset,0]-reducedEvents[i-1,1]<=timeThresh:
                reducedEvents[i-1,1] = temp[i+offset,1]
                offset+=1
            else:
                reducedEvents[i,:] = temp[i+offset,:]
                i+=1
                
        

        
        print offset
        
             
        self.reducedEvents = reducedEvents[:-offset-1,:]
        self.numberOfEvents = size(self.reducedEvents,0)  
            
        return self.reducedEvents
       
        
        
''' Test script
# load in audio to be analyzed
#[x, fs] = M.wavread('/Users/Tlacael/Python/Homework1/mir-noise/feat_extract/RZABR40.wav')

#[x, fs] = M.wavread('wburgShort.wav')
[x, fs] = M.wavread('/Users/tlacael/NYU/MIR/mir-noise/audio_files/GV02_A_Format4min.wav')

run eventDetect.py
#look at segment extraction plot
winLen = fs
hopSize = winLen/2. 


run eventDetect.py
z = onsetDetect(x, fs)


#events = z.findEventLocations()

#events = multiply(events,fs)


run eventDetect.py
winLen = fs
hopSize = winLen/2. 
z = onsetDetect(x, fs)
#peaks = z.envelopeFollowEnergy(winLen,hopSize)
events = z.findEventLocations()

events = multiply(events,fs)


hop = 40
timeX = arange(z.x.size)
timeX.shape = z.x.shape
timeX = divide(timeX, float(fs))
timeX.shape = z.x.shape
plot(timeX[::hop], z.x[::hop], color='r')


#plot envelope
#peaks = divide(peaks,peaks.max())
#envTime = arange(size(peaks))
#nvTime = divide(envTime,fs/float(hopSize))
#envTime.shape = peaks.shape
#plot(envTime,peaks);show()


eventPlot = zeros(size(timeX))
for i in range(size(events,0)):
    eventPlot[events[i,0]:events[i,1]]=0.7
    
plot(timeX[::hop],eventPlot[::hop]);show()

'''


'''


#y = z.envelopeFollow(winLen, hopSize)
y = z.envelopeFollowEnergy(winLen, hopSize)
timeX = arange(z.x.size)
timeX.shape = z.x.shape
timeX = divide(timeX, float(fs))
timeX.shape = z.x.shape
plot(timeX[::40], z.x[::40], color='r')

y= divide(y,y.max())
timeY = range(y.size)
timeY = divide(timeY, (fs/hopSize))
timeY.shape = y.shape
plot(timeY,y, color='b');show()

#w = z.logDeriv()
q = z.pickPeaks(y)
'''
