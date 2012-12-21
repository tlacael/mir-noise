'''
@author: Christopher Jacoby, Tlacael Esparza
MIR Fall 2012: Prof. Juan Bello

Final Project - Feature Extraction & Clustering of Noise Data
'''

import argparse
import numpy as np
import marlib.matlab as M
from datetime import datetime
from mirlib import mir_utils
from mirlib import featurevector
import mirlib.audiofile as af
import mirlib.FFTParams as fftparams
import mirlib.feature_extraction.eventDetect as ed
import mirlib.feature_extraction.calcLoudness as cl
import mirlib.feature_extraction.lowlevel_spectral as llspect
from mirlib.feature_analysis import kmeans
import plot
from matplotlib import pyplot as plt

FEATURE_VECTOR_FILENAME = "features.npy"
SONE_VECTOR_FILENAME = "sones.npy"

def getfeatures(args):
    debug = args.debug
    filepath = args.audiofile
    chunk_len = args.audio_seg_length

    afm = af.audiofile_manager(filepath, chunk_len)

    # FFT Parameters
    fs = afm.afReader.samplerate()
    N = 2048
    hopDenom = 2
    hopSize = N/hopDenom
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
    sone_template = [('sones', 0, 1)]
    # Initialize the feature vector holder
    feature_holder = featurevector.feature_holder(vector_template, filepath)
    sone_holder = featurevector.feature_holder(sone_template, filepath)
    envelopeHolder = []
    audioHolder = []
    sonesHolder = []
    maxEnvelope = 0;
    count =0
    
    print "Feature Extraction Mode\n"
    print datetime.now()
    # For each chunk of audio
    while afm.HasMoreData():
        count +=1
        audioChunk, chunkIndex = afm.GetNextSegment()
        
        
        if debug: print "Read %d sample chunk of audio (%0.2fs)" % (len(audioChunk), len(audioChunk) / fs)

        # Get Events
        eventTimes, envelope = GetEvents(audioChunk, fftParams, debug)
        if maxEnvelope < envelope.max():
            maxEnvelope = envelope.max()
        
        if debug: print "EVENTTIMES:", eventTimes
                
        envelopeHolder.append(envelope)
        
        eventTimesSamps = np.asarray(np.multiply(eventTimes,fs),dtype=int)

        # Get event audio segments
        eventSegments = GetEventAudioSegments(eventTimesSamps, audioChunk, debug)
        
        #get sones
        
        eventSegmentSones = GetEventSones(eventSegments, fftParams, debug)
        
        # Get the MFCCs for each segment / event
        eventSegmentMFCCs = GetEventMFCCs(eventSegments, fftParams, mfccParams, debug)

        # Time-average for each segment / event
        averagedEventSegmentMFCCs, averagedEventSegmentSones = AverageEventFeatures(eventSegmentMFCCs, eventSegmentSones, seglen, fftParams, debug)

        # Store these vectors in the feature_holder, labelled with their time
        StoreFeatureVector(feature_holder, sone_holder, averagedEventSegmentMFCCs, averagedEventSegmentSones, chunkIndex, chunk_len, eventTimes, debug)
        
        if chunkIndex > 16:
            pass
            #break;
    # Write features to disk
    print datetime.now()
    
    fileSize = feature_holder.save(FEATURE_VECTOR_FILENAME)
    print "Wrote", fileSize, "bytes to disk. (%s)" % (FEATURE_VECTOR_FILENAME)

    fileSize = sone_holder.save(SONE_VECTOR_FILENAME)
    print "Wrote", fileSize, "bytes to disk. (%s)" % (FEATURE_VECTOR_FILENAME)

    
def GetEvents(audiodata, fftParams, debug):
    # Get Onsets
    onsetDetector = ed.onsetDetect(fftParams)
        
    # Get Time-Segments from those offsets
    return (onsetDetector.findEventLocations(audiodata))

def GetEventAudioSegments(eventTimes, audiodata, debug):
    ''' eventTimes must be in samples!!! '''
    segments = []
    for i in np.arange(len(eventTimes)):
        segments.append(audiodata[eventTimes[i,0]:eventTimes[i,1]])

        if debug:
            print "\tEvent Detected. Start: %0.2fs, End: %0.2fs, Length: %d samps" % (eventTimes[i,0], eventTimes[i,1], len(segments[i]))

    return segments

def GetEventSones(eventSegments, fftParams, debug):
    soneSegments = []
    for i in np.arange(len(eventSegments)):
        calcSones = cl.SoneCalculator(eventSegments[i], fftParams)
        soneSegments.append(calcSones.calcSoneLoudness())
    return soneSegments

def GetEventMFCCs(eventSegments, fftParams, mfccParams, debug):
    mfccSegments = []
    for i in np.arange(len(eventSegments)):
        X = M.spectrogram(eventSegments[i], fftParams.N, fftParams.h, fftParams.winfunc(fftParams.N))

        #take log magnitude
        
        mfcc = llspect.MFCC_Normalized(X, mfccParams, fftParams)
        mfccSegments.append(mfcc)
        if debug:
            print "\t MFCC:", mfcc.shape
            #print "\t ",mfcc
    return mfccSegments

def AverageEventFeatures(mfccSegments, soneSegments, seglen, fftParams, debug):
    spect_fs = fftParams.fs / fftParams.h
    averaged_mfcc_segs = []
    averaged_sone_segs = []
    for i in np.arange(len(mfccSegments)):
        averaged_segment = mir_utils.AverageFeaturesInTime(mfccSegments[i], spect_fs, seglen)
        averaged_mfcc_segs.append(averaged_segment)

        averaged_segment = mir_utils.AverageFeaturesInTime(soneSegments[i], spect_fs, seglen)
        averaged_sone_segs.append(averaged_segment)
        
    return averaged_mfcc_segs, averaged_sone_segs

def StoreFeatureVector(feature_holder, sone_holder, averagedEventSegmentMFCCs, averagedEventSegmentSones, chunkIndex, chunk_len, eventTimes, debug):
    chunk_start_time = chunkIndex * chunk_len # in seconds
    for i in range(len(averagedEventSegmentMFCCs)):
        chunk_start = eventTimes[i][0]
        chunk_length = eventTimes[i][1] - eventTimes[i][0]
        timekey = (chunk_start_time + chunk_start, chunk_length)
        thissone = averagedEventSegmentSones[i]
        thismfcc = averagedEventSegmentMFCCs[i]
        if debug:
            print "\t  Storing Vector at key:", timekey
        sone_holder.add_feature('sones', thissone, timelabel=timekey)
        feature_holder.add_feature('mfcc', thismfcc, timelabel=timekey)

def clustering(args):
    print "Feature Analysis/Clustering Mode: single k"

    feature_holder = featurevector.feature_holder(filename=FEATURE_VECTOR_FILENAME)
    sones_holder = featurevector.feature_holder(filename=SONE_VECTOR_FILENAME)
    k = args.k

    print feature_holder
    mfccs = feature_holder.get_feature('mfcc')

    print sones_holder
    sones = sones_holder.get_feature('sones')

    '''centroids, nItr = kmeans.kmeans(mfccs, k, thresh)
    print "k-Means with k=%d run in %d iterations." % (k, nItr)'''

    centroids, distortion = Get_Best_Centroids(k, 40)

    
    print "Distortion for this run: %0.3f" % (distortion)

    classes,dist = kmeans.scipy_vq(mfccs, centroids)
    #kmeans.print_vq_stats(mfccs, centroids)

    # Get the inter class dist matrix
    inter_class_dist_matrix = mir_utils.GetSquareDistanceMatrix(centroids)
    print inter_class_dist_matrix

    eventBeginnings = feature_holder.get_event_start_indecies()
    # write audio if given -w
    if args.plot_segments:
        PlotWaveformWClasses(k, feature_holder,classes)
    if args.write_audio_results:
        WriteAudioFromClasses(k, feature_holder, classes)

    plot.plot(mfccs, sones, eventBeginnings, centroids, inter_class_dist_matrix, classes)
    print "J: ", calcJ(mfccs,classes, centroids,k)

def calcJ(mfccs, classes, centroids, k):
    
    Sw = np.zeros((mfccs.shape[1],mfccs.shape[1]))
    Sb = np.zeros((mfccs.shape[1],mfccs.shape[1]))
    
    for i in range(k):
        #sw
        if len(mfccs[(classes == i)]) == 0:
            #print i 
            continue
    
        proportion = np.sum(classes==i)/float(classes.size)
        curClass = mfccs[classes==i]
        
        if np.ndim(curClass) ==1:
            curClass.shape = (1,curClass.size)
            covar = np.outer(curClass,curClass)
            Sw += np.multiply(proportion,covar)
        elif len(curClass) == 1:
            covar = np.outer(curClass,curClass)
            Sw += np.multiply(proportion,covar)
        else:
            covar = np.cov(curClass.T)
            Sw += np.multiply(proportion,covar)

        #Sb
        globalMean = np.mean(mfccs, 0)
        meanOfClass = np.mean(mfccs[classes==i],0)
        diff = meanOfClass - globalMean

        Sb += np.multiply(np.outer(diff,diff), proportion)

    SWsumDiag = sum(np.diag(Sw))
    SBsumDiag = sum(np.diag(Sb))

    return SBsumDiag/SWsumDiag
    
def feature_selection(args):
    print "Feature Analysis/Clustering Mode - feature selection from multiple k's"

    feature_holder = featurevector.feature_holder(filename=FEATURE_VECTOR_FILENAME)
    kMin = args.k_min
    kMax = args.k_max
    kHop = args.k_hop

    mfccs = feature_holder.get_feature('mfcc')

    results = []
    j_measures = np.zeros(kMax- kMin)
    
    for k in range(kMin, kMax, kHop):
        centroids, distortion = Get_Best_Centroids(k,20)

        classes, dist = kmeans.scipy_vq(mfccs, centroids)

        j_measures[(k-kMin)/kHop] = calcJ(mfccs, classes, centroids, k)
        results.append( (k, distortion, dist) )

    #print [ (a) for (a,b,c) in results]

    print "jMeasures", j_measures
    plt.close()
    plt.plot(j_measures);plt.show()
    #print [ (a, b) for (a,b,c) in results]
    
def Get_Best_Centroids(k, iterations):

    feature_holder = featurevector.feature_holder(filename=FEATURE_VECTOR_FILENAME)
    
    mfccs = feature_holder.get_feature('mfcc')

    j_measures = np.zeros(iterations)
    max = 0;
    bestCentroids = 0
    bestDistortion = 0
    
    for i in range(iterations):
        centroids, distortion = kmeans.scipy_kmeans(mfccs, k)

        classes, dist = kmeans.scipy_vq(mfccs, centroids)

        j_measures[i] = calcJ(mfccs, classes, centroids, k)

        if j_measures[i] > max:
            max = j_measures[i]
            bestCentroids = centroids
            bestDistortion = distortion

    '''
    plt.close()
    fig, (ax1) = plt.subplots(1)
    ax1.plot(j_measures)
    ax1.set_title("J measures over multiple iterations of k")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("J-measure values")
    plt.show()
    '''
    return bestCentroids, bestDistortion
    
def PlotWaveformWClasses(k, feature_holder, classes):
    index_time_map = feature_holder.get_index_time_map()
    print "Original File:", feature_holder.filename

    
    # for each class k
    events = np.zeros((60*6*44100,1))
    segment_classes = GetClassFromSegment(k, index_time_map, classes)
    for i in sorted(segment_classes.keys()):
        # Find all time segments that go with this class
        timeSegments = [ index_time_map[j] for j  in sorted(segment_classes[i])]
        
        
        
        # Write all these time segments to a single file
        audioSegments, fs = af.get_arbitrary_file_segments(feature_holder.filename, timeSegments)
        
        for j in range(len(timeSegments)):
            print j, timeSegments[j]
            if (timeSegments[j][0]+timeSegments[j][1]) < 60*6:
                
                startIndex = int(timeSegments[j][0]*fs)
                endIndex = int(float(timeSegments[j][0]+timeSegments[j][1])*fs)
                
                diff = len(audioSegments[j])-len(events[startIndex:endIndex])
                events[startIndex:endIndex+diff] = audioSegments[j]
                
    plt.plot(events[::100]);plt.show()
    
        
    
def WriteAudioFromClasses(k, feature_holder, classes):
    index_time_map = feature_holder.get_index_time_map()
    print "Original File:", feature_holder.filename

    # for each class k
    segment_classes = GetClassFromSegment(k, index_time_map, classes)
    for i in sorted(segment_classes.keys()):
        # Find all time segments that go with this class
        timeSegments = [ index_time_map[j] for j  in sorted(segment_classes[i])]

        print timeSegments
        # Write all these time segments to a single file
        audioSegments, fs = af.get_arbitrary_file_segments(feature_holder.filename, timeSegments)
        resultDir = './results'
        af.write_segment_audio("%s/class-%d.wav" % (resultDir, i), audioSegments, fs)
    
def GetClassFromSegment(k, index_time_map, classes):
    results = {}
    for time_seg in sorted(index_time_map.keys()):
        start = time_seg[0]
        end = start + time_seg[1]
        time_seg_classes = classes[ start : end ]
        hist, edges = np.histogram( time_seg_classes, np.arange(k) )
        segment_class = np.argmax(hist)
        if results.has_key(segment_class):
            results[segment_class].append(time_seg)
        else:
            results[segment_class] = [time_seg]

    return results

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

    # Parameters for Feature Selection
    parser_featureselection = subparsers.add_parser("featureselect", help="Feature Selection Mode")
    parser_featureselection.add_argument("k_min", default=2, type=int)
    parser_featureselection.add_argument("k_max", default=100, type=int)
    parser_featureselection.add_argument("k_hop", default=5, type=int)
    parser_featureselection.set_defaults(func=feature_selection)

    # Parameters for Clustering mode
    parser_clustering = subparsers.add_parser("clustering", help="Feature Analysis Mode")
    parser_clustering.add_argument("k", help="Number of classes", type=int)
    parser_clustering.add_argument("-w", "--write_audio_results", help="Write audio from clusters", action="store_true")
    parser_clustering.add_argument("-plot", "--plot_segments", help="Plot audio segment classes", action="store_true")
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
