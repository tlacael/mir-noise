import os
import numpy as np
import marlib.matlab as M
import mirlib.feature_extraction.eventDetect as ed
import mirlib.FFTParams as fftparams
from matplotlib.pylab import *

inputfile = '../audio/GV02_A_Format4min.wav'

if not os.path.exists(inputfile):
    raise Exception("FILE DOES NOT EXIST, TRY AGAIN")

[x, fs] = M.wavread(inputfile)

# FFT Parameters
N = 2048
hopDenom = 2
zp = 0
winfunc=np.hamming
fftParams = fftparams.FFTParams(fs, N, hopDenom, zp, winfunc)

#peaks = z.envelopeFollowEnergy(winLen,hopSize) # the old way
z = ed.onsetDetect(fftParams)
events = z.findEventLocations(x)

events = np.multiply(events,fs)

hop = 100
timeX = arange(x.size)
timeX.shape = x.shape
timeX = np.divide(timeX, np.float(fs))
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


eventPlot = np.zeros(size(timeX))
for i in range(size(events,0)):
    eventPlot[events[i,0]:events[i,1]]=0.7
    
ax1.plot(timeX[::hop],eventPlot[::hop])
print timeX.shape, eventPlot.shape, timeX[::hop].shape, eventPlot[::hop].shape
print "show..."

show()
