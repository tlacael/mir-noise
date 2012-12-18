import os
import numpy as np
import marlib.matlab as M
import mirlib.feature_extraction.eventDetect as ed
import mirlib.FFTParams as fftparams
from matplotlib.pylab import *


#inputfile = '../audio_files/GV02_A_Format4min.wav'
inputfile = '../audio_files/wburgShort.wav'

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

xConcat = x[(events)]


M.wavwrite(xConcat, "segs.wav", fs)