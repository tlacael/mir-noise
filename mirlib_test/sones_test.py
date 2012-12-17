
from numpy import *
from matplotlib.pyplot import *
import marlib.matlab as M
import scipy as sp
import math

import mirlib.FFTParams as fftparams
import mirlib.feature_extraction.calcLoudness as cl


[x,fs] = M.wavread('../audio_files/WB_12-15_342pm10mins.wav')
winLen = 4096*8
hopSize = winLen
# FFT Parameters
N = 4096*8
hopDenom = 1.
hopSize = N/float(hopDenom)
zp = 0
winfunc=np.hamming
fftParams = fftparams.FFTParams(fs, N, hopDenom, zp, winfunc)



#[x,fs] = M.wavread('RZABR40.wav')
#[x,fs] = M.wavread('RZABR40pad.wav')

z = cl.SoneCalculator(x, fftParams)
sonVec = z.calcSoneLoudness()

close()

fig = figure()
ax1 = fig.add_subplot(111)
signalTime = arange(z.y.size)
signalTime.shape = z.y.shape
hop = 60
signalTime = divide(signalTime,(60*float(fs)))
ax1.plot(signalTime[::hop], z.y[::hop],color='c', label="Original Waveform")

#sonVec= divide(sonVec,sonVec.max())
timeY = range(sonVec.size)
timeY = divide(timeY, (60*(z.fs/float(hopSize))))
timeY.shape = sonVec.shape

ax1.set_title("Sones Measurement")
ax1.set_ylim(-1, 1)
ax1.set_xlabel("Minutes")

sonVec = divide(sonVec,max(sonVec))
ax1.plot(timeY,sonVec, color='r', label="Sones Curve")
ax1.legend()
show()