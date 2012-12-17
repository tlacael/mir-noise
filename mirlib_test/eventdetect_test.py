import os
import numpy as np
import marlib.matlab as M
import mirlib.feature_extraction.eventDetect as ed
import mirlib.FFTParams as fftparams
from matplotlib.pylab import *

#inputfile = '../audio_files/GV02_A_Format4min.wav'
inputfile = '../audio_files/WB_12-15_342pm10mins.wav'

if not os.path.exists(inputfile):
    raise Exception("FILE DOES NOT EXIST, TRY AGAIN")

[x, fs] = M.wavread(inputfile)

# FFT Parameters
N = 4096
hopDenom = 2
hopSize = N/float(hopDenom)
zp = 0
winfunc=np.hamming
fftParams = fftparams.FFTParams(fs, N, hopDenom, zp, winfunc)

#peaks = z.envelopeFollowEnergy(winLen,hopSize) # the old way
z = ed.onsetDetect(fftParams)


events, envelope = z.findEventLocations(x)



makePlot = 1


if makePlot:
    events = np.multiply(events,4)
    
    hop = 100
    timeX = arange(x.size)
    timeX.shape = x.shape
    timeX = np.divide(timeX, np.float(fs)*60)
    timeX.shape = x.shape
    print "Plotting..."
    fig = figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(timeX[::hop], x[::hop], color='r')
    
    #plot envelope
    #peaks = divide(peaks,peaks.max())
    #envTime = arange(size(peaks))
    #nvTime = divide(envTime,fs/float(hopSize))
    #envTime.shape = peaks.shape
    #plot(envTime,peaks);show()
    
    envelope = envelope[:size(x)/(fs/4.)]
    if 1:
        eventPlot = np.zeros(size(envelope))
        for i in range(size(events,0)):
            
            eventPlot[events[i,0]:events[i,1]]=0.9
        
    envelope = eventPlot * envelope    
      #  ax1.plot(timeX[::hop],eventPlot[::hop])
      #  print timeX.shape, eventPlot.shape, timeX[::hop].shape, eventPlot[::hop].shape
       # print "show..."
    
    
    #events = array(np.divide(events, (fs/2)),dtype=int)
    timeEnv = np.divide((range(size(envelope))) , 60*4.)
    envelope = np.divide(envelope,0.5*envelope.max())
    ax1.set_title("Event Segmenting w/ Energy Envelope")
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Minutes")
    ax1.plot(timeEnv, eventPlot)
    ax1.plot(timeEnv, envelope)
    show()

