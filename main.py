'''
@author: Christopher Jacoby, Tlacael Esparza
MIR Fall 2012: Prof. Juan Bello

Final Project - Feature Extraction & Clustering of Noise Data
'''

import argparse
import numpy as np
import marlib.matlab as M

from mirlib import mir_utils
from mirlib import featurevector
import mirlib.audiofile as af
import mirlib.FFTParams as fftparams
import mirlib.feature_extraction.eventDetect as ed
import mirlib.feature_extraction.lowlevel_spectral as llspect
from mirlib.feature_analysis import kmeans
import plot

FEATURE_VECTOR_FILENAME = "features.npy"

def getfeatures(args):
    debug = args.debug
    filepath = args.audiofile
    chunk_len = args.audio_seg_length

    afm = af.audiofile_manager(filepath, chunk_len)

    # FFT Parameters
    fs = afm.afReader.samplerate()
    N = 2048
    hopDenom = 2
    zp = 0
    winfunc=np.hamming
    fftParams = fftparams.FFTParams(fs, N, hopDenom, zp, winfunc)

    # MFCC Paramters
    nFilters = 40
    nDCTCoefs = 20
    minFreq = 50
    maxFreq = 8000
    nIndexSkip = 2
    seglen = 1
    mfccParams = fftparams.MFCCParams(nFilters, nDCTCoefs, minFreq, maxFreq, nIndexSkip)

    # Feature Vector parameters
    # Template : ('name', order index, length)
    vector_template = [('sones', 0, 1),
                        ('mfcc', 1, nDCTCoefs - nIndexSkip)]
    # Initialize the feature vector holder
    feature_holder = featurevector.feature_holder(vector_template)
    
    print "Feature Extraction Mode\n"
    # For each chunk of audio
    while afm.HasMoreData():
        audioChunk, chunkIndex = afm.GetNextSegment()

        if debug: print "Read %d sample chunk of audio (%0.2fs)" % (len(audioChunk), len(audioChunk) / fs)

        # Get Events
        eventTimes = GetEvents(audioChunk, fs, debug)
        if debug: print "EVENTTIMES:", eventTimes
        eventTimesSamps = np.asarray(np.multiply(eventTimes,fs),dtype=int)

        # Get event audio segments
        eventSegments = GetEventAudioSegments(eventTimesSamps, audioChunk, debug)

        # Get the MFCCs for each segment / event
        eventSegmentMFCCs = GetEventMFCCs(eventSegments, fftParams, mfccParams, debug)

        # Time-average for each segment / event
        averagedEventSegmentMFCCs = AverageEventMFCCs(eventSegmentMFCCs, seglen, fftParams, debug)

        # Store these vectors in the feature_holder, labelled with their time
        StoreFeatureVector(feature_holder, averagedEventSegmentMFCCs, chunkIndex, chunk_len, eventTimes, debug)

        if chunkIndex > 8:
            break;
        
    # Write features to disk
    fileSize = feature_holder.save(FEATURE_VECTOR_FILENAME)
    print "Wrote", fileSize, "bytes to disk."

def GetEvents(audiodata, fs, debug):
    # Get Onsets
    onsetDetector = ed.onsetDetect(audiodata,fs)
        
    # Get Time-Segments from those offsets
    return onsetDetector.findEventLocations()

def GetEventAudioSegments(eventTimes, audiodata, debug):
    ''' eventTimes must be in samples!!! '''
    segments = []
    for i in np.arange(len(eventTimes)):
        segments.append(audiodata[eventTimes[i,0]:eventTimes[i,1]])

        if debug:
            print "\tEvent Detected. Start: %0.2fs, End: %0.2fs, Length: %d samps" % (eventTimes[i,0], eventTimes[i,1], len(segments[i]))

    return segments

def GetEventMFCCs(eventSegments, fftParams, mfccParams, debug):
    mfccSegments = []
    for i in np.arange(len(eventSegments)):
        X = M.spectrogram(eventSegments[i], fftParams.N, fftParams.h, fftParams.winfunc(fftParams.N))

        X = np.array(X)       
        XmagDB = 20*np.log10(abs(X)) #take log magnitude
        
        mfcc = llspect.MFCC_Normalized(XmagDB, mfccParams, fftParams)
        mfccSegments.append(mfcc)
        if debug:
            print "\t MFCC:", mfcc.shape
            #print "\t ",mfcc
    return mfccSegments

def AverageEventMFCCs(mfccSegments, seglen, fftParams, debug):
    spect_fs = fftParams.fs / fftParams.h
    averaged_mfcc_segs = []
    for i in np.arange(len(mfccSegments)):
        averaged_segment = mir_utils.AverageFeaturesInTime(mfccSegments[i], spect_fs, seglen)
        averaged_mfcc_segs.append(averaged_segment)
    return averaged_mfcc_segs

def StoreFeatureVector(feature_holder, averagedEventSegmentMFCCs, chunkIndex, chunk_len, eventTimes, debug):
    chunk_start_time = chunkIndex * chunk_len # in seconds
    for i in range(len(averagedEventSegmentMFCCs)):
        chunk_start = eventTimes[i][0]
        chunk_length = eventTimes[i][1] - eventTimes[i][0]
        timekey = (chunk_start_time + chunk_start, chunk_length)
        thismfcc = averagedEventSegmentMFCCs[i]
        if debug:
            print "\t  Storing Vector at key:", timekey
        feature_holder.add_feature('mfcc', thismfcc, timelabel=timekey)

def clustering(args):
    print "Feature Analysis/Clustering Mode"

    feature_holder = featurevector.feature_holder(filename=FEATURE_VECTOR_FILENAME)
    k = args.k
    thresh = args.stop_threshold

    print feature_holder
    mfccs = feature_holder.get_feature('mfcc')

    print feature_holder.vector.shape
    centroids, nItr = kmeans.kmeans(mfccs, k, thresh)
    print "k-Means with k=%d run in %d iterations." % (k, nItr)
    print centroids
    
    plot.plot(mfccs, centroids)

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
    parser_clustering.add_argument("k", help="Number of classes", type=int)
    parser_clustering.add_argument("-T", "--stop_threshold", help="Threshold to stop iterating k-means", default=0.01, type=float)
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
