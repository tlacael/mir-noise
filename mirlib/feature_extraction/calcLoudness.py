''' MIR Final Project - City Sounds Categorization
Author Tlacael Esparza'''

''' Class to calculate loudness vector of signal with window winLen
arguments are filename of wav audio to be analyzed and analysis window size
usage: soneVec = calcLoudness(filename, winLength)'''


from numpy import *
from matplotlib.pyplot import *
import marlib.matlab as M
import scipy as sp
import math
import eventDetect as ED

'''
run calcLoudness.py
[x,fs] = M.wavread('wburgShort.wav')
#[x,fs] = M.wavread('RZABR40.wav')
#[x,fs] = M.wavread('RZABR40pad.wav')
run calcLoudness.py
winLen = 4096*8
hopSize = winLen
z = SoneCalculator(x,fs, winLen)
sonVec = z.calcSoneLoudness()

close()

signalTime = arange(z.y.size)
signalTime.shape = z.y.shape
hop = 40
signalTime = divide(signalTime,float(fs))
plot(signalTime[::hop], z.y[::hop],color='r')

#sonVec= divide(sonVec,sonVec.max())
timeY = range(sonVec.size)
timeY = divide(timeY, (z.fs/float(hopSize)))
timeY.shape = sonVec.shape
plot(timeY,sonVec, color='b');show()


'''

class SoneCalculator:
    def __init__(self, y, fs, winLen=1024):
        
                
        if y.ndim == 2 and y.shape[1] == 1:
            y.shape = (y.shape[0]) 
        self.y = y;
        self.fs = fs;
        self.winLen = winLen


    def calcSoneLoudness(self):     
        
        yPad = zeros((self.winLen/2))
        yPad = concatenate((yPad, self.y, yPad),0)
        
        self.yBuf = M.shingle(yPad, self.winLen, self.winLen)
        self.bufLen = size(self.yBuf, 0)

        soneVec = zeros(self.bufLen)

        self.count = 0
        for i in range(self.bufLen):
            self.count +=1
            soneVec[i] = self.getSegmentSone(i)

        self.S = soneVec
        
        #gate signal
        E = ED.onsetDetect(self.y, self.fs)
        envelope = E.envelopeFollow(self.winLen*2, self.winLen)
        envelope = divide(envelope,envelope.max())
        
        soneVec = soneVec * envelope
        
        return soneVec


    ''' Sone measurement'''

    def getSegmentSone(self, i):
        
        y = self.yBuf[i,:]
        self.ySig = y

        SPL_meas = 70.
        presRef = 2e-5
        y_scaled = divide(y, presRef)
        RMS = sqrt(mean(square(y_scaled)))

        SPL = multiply(20, log10(RMS))

        calib = pow(10, divide(subtract(SPL_meas, SPL),20))
        y_calib = multiply(calib, y_scaled)
        
        
        
        self.RMS_SELF = RMS
        self.SPL_MAT = SPL_meas
        self.CALIB = calib
        self.YSCALED = y_scaled
        self.yCalib = y_calib
        
        '''direct method of calculating sone'''
        
        '''set params '''
        NFFT = 1024#self.winLen
        ''' barks one to 18'''
    
        barkFreqs = range(1,19)    
        
        y_calib = y_calib.reshape(size(y_calib))
        [Ypsd, powFreqs] = psd(y_calib, NFFT, self.fs, Fc=0, window=hanning(NFFT), noverlap=0 )
        self.psdFreqs = powFreqs

        Y_scaled = divide(multiply(2,Ypsd),NFFT)
        
        B_bands = self.calcBarkScale(barkFreqs, powFreqs, Y_scaled)

        crit_sprd = self.calcCritSpread(barkFreqs, B_bands)

        phons = self.dBtoPhon(crit_sprd)

        sones = self.PhonsToSones(phons)
        
        sumMBSD = sum(sones)
    
        return sumMBSD
        
        
    ''' function to calculate critical bands in the bark sprectrum'''
    ''' critical bands from "foundation of modern auditory theory [insert cite]'''
    def calcBarkScale(self, barkFreqs, powFreqs, Y_scale):
        
        bark = [0,100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720\
        ,2000, 2320, 2700, 3150, 3700, 4400]
        
        B_bands = zeros(18)
        for i in range(2,19):
            B_bands[i-2] = sum( Y_scale[logical_and(bark[i-2] <= powFreqs, powFreqs<bark[i-1])])
            
        return B_bands
    
    
            
    def calcCritSpread(self, barkFreqs, B_bands):
    
        
        spread = zeros([size(barkFreqs),size(barkFreqs)])
        for i in barkFreqs:
            for j in barkFreqs:
                powVal = (15.81+7.5 * ((i-j)+0.474)-17.5 * pow(pow(1+((i-j)+0.474),2),0.5))/10
                spread[i-1,j-1] = pow(10,powVal)
                
        self.crit_sprd = dot(spread,B_bands[:,np.newaxis])
        
        self.sBread = spread
        self.bands = B_bands
        
        return  self.crit_sprd
    
    def dBtoPhon(self, crit_sprd): 
        eqlcon = \
        matrix('12,7,4,1,0,0,0,-0.5,-2,-3,-7,-8,-8.5,-8.5,-8.5;\
        20,17,14,12,10,9.5,9,8.5,7.5,6.5,4,3,2.5,2,2.5;\
        29,26,23,21,20,19.5,19.5,19,18,17,15,14,13.5,13,13.5;\
        36,34,32,30,29,28.5,28.5,28.5,28,27.5,26,25,24.5,24,24.5;\
        45,43,41,40,40,40,40,40,40,39.5,38,37,36.5,36,36.5;\
        53,51,50,49,48.5,48.5,49,49,49,49,48,47,46.5,45.5,46;\
        62,60,59,58,58,58.5,59,59,59,59,58,57.5,57,56,56;\
        70,69,68,67.5,67.5,68,68,68,68,68,67,66,65.5,64.5,64.5;\
        79,79,79,79,79,79,79,79,78,77.5,76,75,74.5,73,73;\
        89,89,89,89.5,90,90,90,89.5,89,88.5,87,86,85.5,84,83.5;\
        100,100,100,100,100,99.5,99,99,98.5,98,96,95,94.5,93.5,93;\
        112,112,112,112,111,110.5,109.5,109,108.5,108,106,105,104.5,103,102.5;\
        122,122,121,121,120.5,120,119,118,117,116.5,114.5,113.5,113,111,110.5')
    
        phonList = matrix('0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0')
        
        self.T = multiply(10,log10(self.crit_sprd[3:18]))
        self._phons = zeros(15)
        
        for i in range(0,15):
            j = 0
            while self.T[i] >= eqlcon[j,i]:
                j+=1
            
                if j==0:
                   self._phons[i] = phonList[0,0]
                else:
                    t1 = divide((self.T[i] - eqlcon[j-1,i]),(eqlcon[j,i] - eqlcon[j-1,i]));
                    self._phons[i] = phonList[0,j-1] + t1*(phonList[0,j]- phonList[0,j-1])
            
        return self._phons
            

    def PhonsToSones(self, phons):
        sones = zeros(15)
        for i in range(0,15):
            if phons[i] >= 40:
                sones[i] = pow(2,(phons[i] - 40)/10)
            else:
                sones[i] = pow(phons[i]/40,2.642)
            
        return sones
        
        

    
    
