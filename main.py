'''
@author: Christopher Jacoby, Tlacael Esparza
MIR Fall 2012: Prof. Juan Bello

Final Project - Feature Extraction & Clustering of Noise Data
'''

import argparse
import mirlib.audiofile as af
import mirlib.feature_extraction.eventDetect as ed
import numpy as np
import mirlib.FFTParams as fftparams
from mirlib import mir_utils
from mirlib import featurevector
import marlib.matlab as M

import mirlib.feature_extraction.lowlevel_spectral as llspect

def getfeatures(args):
    filepath = args.audiofile
    audio_seg = args.audio_seg_length

    afm = af.audiofile_manager(filepath, audio_seg)
    fs = afm.afReader.samplerate()

    # FFT Parameters
    fs = afm.afReader.samplerate()
    N = 1024
    hopDenom = 2
    zp = 0
    winfunc=np.hamming
    fftParams = fftparams.FFTParams(fs, N, hopDenom, zp, winfunc)

    # MFCC Paramters
    nFilters = 40
    nDCTCoefs = 20
    minFreq = 50
    maxFreq = 8000
    nIndexSkip = 0
    seglen = 1

    # Feature Vector parameters
    # Template : ('name', order index, length)
    vector_template = [('sones', 0, 1),
                        ('mfcc', 1, nDCTCoefs)]
    feature_holder = featurevector.featurevector_holder(vector_template)
    
    print "Feature Extraction Mode"
    # For each chunk of audio
    while afm.HasMoreData():
        audioChunk, chunkIndex = afm.GetNextSegment()
        # Only look at Left if there's more than that
        if (audioChunk.shape) > 1 and audioChunk.shape[1] > 1:
            audioChunk = audioChunk[:,0]

        print "Read %d sample chunk of audio" % (len(audioChunk))
        
        # 1. Get Onsets
        s = ed.onsetDetect(audioChunk,fs)
        
        # 2. Get Time-Segments from those offsets
        segmentTimes = s.findEventLocations()
        segmentTimesSamps = np.asarray(np.multiply(segmentTimes,fs),dtype=int)
        segments = []
        for i in np.arange(np.size(segmentTimes,0)):
            segments.append(audioChunk[segmentTimesSamps[i,0]:segmentTimesSamps[i,1]])
            
            print "Event Detected. Start: %0.2fs, End: %0.2fs, Length: %d samps" % (segmentTimes[i,0],segmentTimes[i,1],len(segments[i]))

        # 3. Get the MFCCs for each segment / event
        '''for i in np.arange(len(segmentTimes)):
            X = M.spectrogram(segments[i], N, N / hopDenom, winfunc(N))
            spect_fs = fs / (N / hopDenom)
        
            mfcc = llspect.MFCC_Normalized(X, nFilters, nDCTCoefs, minFreq, maxFreq, fftParams)
            #print mfcc.shape
            #print mfcc
            ##featureDict[index] = mfcc
        '''

        # 4. Time-average for each segment / event
        #time_averaged_mfcc = mir_utils.AverageFeaturesInTime(mfcc, spect_fs, seglen)

        # 5. Write to disk
        #if index > 5:
        #    break

    '''i = 0
    mfccResult = featureDict[i]
    i += 1
    while i in featureDict.keys():
        mfccResult = np.concatenate([mfccResult, featureDict[i]])
        i += 1

    from matplotlib.pylab import *
    figure()
    imshow(mfcc_norm.T, interpolation='nearest', origin='lower', aspect='auto')
    show()'''

def clustering(args):
    print "Feature Analysis/Clustering Mode"

def ParseArgs():
    ''' Parse the program arguments & run the appropriate functions '''

    parser = argparse.ArgumentParser()
    # Main program parameters
    parser.add_argument("-d", "--debug", action="store_true")

    subparsers = parser.add_subparsers(help="Program Mode Help")
    # Parameters for Feature Extraction
    parser_getfeatures = subparsers.add_parser("getfeatures", help="Feature Extraction Mode")
    parser_getfeatures.set_defaults(func=getfeatures)
    parser_getfeatures.add_argument("audiofile", help="Input audio file path")
    parser_getfeatures.add_argument("-l", "--audio_seg_length", help="Amount of audio data to process at a time", default=30, type=int)

    # Parameters for Clustering mode
    parser_clustering = subparsers.add_parser("clustering", help="Feature Analysis Mode")
    parser_clustering.set_defaults(func=clustering)

    args = parser.parse_args()
    args.func(args)

def main():
    ''' handle argument parsing '''

    print 'Christopher Jacoby & Tlacael Esparza'
    print 'MIR-Noise'
    print

    ParseArgs()

if __name__ == '__main__':
    main()
