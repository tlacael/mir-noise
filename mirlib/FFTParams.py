import numpy as np

class FFTParams:
    def __init__(self, samplingFreq=44100, N=512, hopDenom=2, zp=0, winfunc=np.ones):
        self.fs = samplingFreq
        self.N = N
        self.h = N / hopDenom
        self.zeropad = zp
        self.winfunc = winfunc
