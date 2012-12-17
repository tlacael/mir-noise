''' @author: Tlacael Esparza'''
''' script analyzes wav audio and detects events 
with novelty function, returns index with beginning and ending time markings 
for identified event'''


from numpy import *
from matplotlib.pyplot import *
import marlib.matlab as M
import scipy as sp
import math
import lowlevel_spectral as llspect

class onsetDetect:
    
    def __init__(self, fftParams):
            
        self.fftParams = fftParams        
        self.fs = fftParams.fs
        
    def envelopeFollowEnergy(self, x, winLen, hopSize):
        self.winLen = winLen
        self.hopSize = hopSize
        
        xPad = zeros(self.winLen/2)

       # size(xPad,1) = size(self.x,1)
        xPad = concatenate([xPad, x, xPad])
        
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
    
    def GetTimeEnvelope(self, x):
        
        # pad the beginning and end with zeros so we fix time issues
        xPad = zeros(self.fftParams.N/2)
        xPad = concatenate([xPad, x, xPad])
        
        X = M.spectrogram(xPad, self.fftParams.N, self.fftParams.h, self.fftParams.winfunc(self.fftParams.N))
        X = np.array(X)

        return llspect.SpectralFlux(X, self.fftParams, 1)

    def SmoothEnvelope(self, envelope, window_len, hop_size):
        xPad = zeros(window_len/2)

       # size(xPad,1) = size(self.x,1)
        xPad = concatenate([xPad, envelope, xPad])

        # Iterate over xPad and smooth with a window
        xBuf = M.shingle(xPad, window_len, hop_size)

        xBufSmoothed = median(xBuf, axis=1)
        
        return xBufSmoothed
        
        
    def findEventLocations(self, x):
        # Do this to make sure it works for both our machines.
        # sometimes x is (n, 1) and sometimes x is (n,). Dunno why.
        if x.ndim == 2 and x.shape[1] == 1:
            x.shape = (x.shape[0])

        spect_fs = self.fftParams.fs / np.float(self.fftParams.N)

        #set 2 second window for strong smoothing
        smoothingWinLen = self.fftParams.fs*0.5
        smoothingHopSize = smoothingWinLen/2.
        
        #using spectral flux
        #self.envelope = self.GetTimeEnvelope(x)

        #using energy envelope
        envelope = self.envelopeFollowEnergy(x, smoothingWinLen, smoothingHopSize)


        # CBJ : Do we really need smoothing now? and if so, by how much? we're already
        # getting a 2048 sample value, which is about .05s. We'd only need like 10 of these
        # to get a half second of smoothing; more than that is not really useful at this stage;
        # we'd loose too much resolution on events.
        #self.envelope = self.SmoothEnvelope(envelope, smoothingWinLen, smoothingHopSize)
        #return envelope, self.envelope
        
        #normalize
        #self.envelope = divide(self.envelope, self.envelope.max()) 
        

        #EnvThresh = envelope           
        EnvThresh = envelope.copy(0)   
        thresh = 0.0007#median(EnvThresh)

        
        EnvThresh[EnvThresh<thresh]=0
        
        EventCenters = nonzero(EnvThresh)
   
        EventCenters = array(EventCenters)
    
        lengths = zeros((EventCenters.size,2))
        count = 0

        
        #len=0  # len is probably not a good name for a variable, since it's a built-in function
        
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
        
        self.numberOfEvents = EventCenters.size
        
        EventTimes = zeros((self.numberOfEvents,2))
        eventIndex = arange(self.numberOfEvents)
        
        chunkLen = x.size/self.fs
        print "signal length", chunkLen
        print "signal size", x.size
        
        
        #convert event centers to time windows in seconds
        
        widen = 0.2 #amount to pad window on either side of event, in seconds
        padAdjust = 0.5

        for i in eventIndex:
            EventTimes[i,0] = (EventCenters[0,i] - 0.5 - padAdjust)*smoothingHopSize/float(self.fs)-widen
            if EventTimes[i,0] < 0:
                EventTimes[i,0] = 0
            EventTimes[i,1] = (EventCenters[0,i] + 0.5- padAdjust)*smoothingHopSize/float(self.fs)+widen
            if EventTimes[i,1] > chunkLen:
                EventTimes[i,1] = chunkLen
            
        #make sure time vlues not out of bounds
        self.EventTimes = EventTimes
        
        
        eventIndex = arange(self.numberOfEvents)
        #reduce number of segments i.e. overlaps, nearby
        reducedEvents = zeros((self.numberOfEvents,2))
        temp = self.EventTimes
        
        offset =0;
        timeThresh = 0.02#1*winLen/float(self.fs) #in seconds
        i = 1
        
        
        
        if self.numberOfEvents > 0:
            reducedEvents[0,:] = temp[0,:]
        
        while((i +offset) < self.numberOfEvents):
            if temp[i+offset,0]-reducedEvents[i-1,0] <=0:
                reducedEvents[i-1,1] = temp[i+offset,1]
                offset+=1
            elif temp[i+offset,0]-reducedEvents[i-1,1]<=timeThresh:
                reducedEvents[i-1,1] = temp[i+offset,1]
                offset+=1
            else:
                reducedEvents[i,:] = temp[i+offset,:]
                i+=1
                
        reducedEvents = reducedEvents[:-offset,:]
        self.numberOfEvents = size(reducedEvents,0)  
        print "number of events:", self.numberOfEvents
            
        return (reducedEvents, envelope)

    '''
        #THIS CODE DOES NOT SEEM TO BE IN USE. TRUE?
        
    def envelopeFollow(self, winLen, hopSize):
        self.winLen = winLen
        self.hopSize = hopSize
    
        xPad = zeros((self.winLen/2))

      #  size(xPad,1) = size(x,1)
        xPad = concatenate([xPad, x])
            
        
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

    def logDeriv(self):

        xLogDeriv = zeros(size(self.xLocEnrg))
        
        xLogDeriv[0:-1] = log(self.xLocEnrg[1:]) - log(self.xLocEnrg[0:-1])
        xLogDeriv[-1] = 0
        
        xLogDeriv[abs(self.xLocEnrg) < mean(abs(self.xLocEnrg))] = 0
         
        self.xLogDeriv = xLogDeriv
        
        return self.xLogDeriv
'''    
        
        
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
