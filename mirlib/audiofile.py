'''
@author Christopher Jacoby

Routines for handling high level audio file management.
'''


import os
import glob
import marlib.audiofile as af
import marlib.matlab as M
import numpy as np

RESULT_DIR = './segments'

def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def cleandir(dir):
    if os.path.exists(dir):
        cleanglob = "%s/*" % (dir)
        files = glob.glob(cleanglob)
        for f in files:
            os.remove(f)

class audiofile_manager:
    ''' Class for loading large audio files in small chunks. '''
    
    def __init__(self, filepath, segment_length):
        self.filepath = filepath
        self.filename = os.path.split(filepath)[1]
        self.index = 0

        self.afReader = af.AudioReader(filepath)
        self.seg_length_samps = int(np.round(segment_length * self.afReader.samplerate()))

    def HasMoreData(self):
        ''' Returns true if there are more segments available in the audio file '''
        return (self.index * self.seg_length_samps) < self.afReader.numsamples()

    def GetNextSegment(self, bMono=True):
        ''' Returns the samples for the next available audio segment.
        bMono=false returns it as a (N,2) shape, bMono=true returns it as (N)'''
        segment = self.afReader.read_frame_at_index(self.index * self.seg_length_samps, self.seg_length_samps)
        self.index += 1

        if bMono and (segment.shape) > 1 and segment.shape[1] > 1:
            segment = segment[:, 0]
        
        return segment, (self.index - 1)


def segment_audio_files(filepath, segment_length, result_dir=RESULT_DIR):
    ''' From an input audio file, segments the file into segments of length segment_length, and outputs them as individual files to result_dir
    Parameters:
       filepath - the path to the input file
       segment_length - the length of the segment to return, in seconds
       result_dir - output path to write to '''
    filename = os.path.split(filepath)[1]
    afm = audiofile_manager(filepath, segment_length)

    checkdir(RESULT_DIR)
    cleandir(RESULT_DIR)
    
    print "File in 5-minute chunks"
    while afm.HasMoreData():
        segment, index = afm.GetNextSegment()

        outFile = "%s/%s-%d.wav" % (RESULT_DIR, filename[:-4], index)
        print "%d %.3f%%" % (index, (index * afm.seg_length_samps / np.float(afm.afReader.numsamples()))*100  )
        print "Writing %s..." % (outFile),
        M.wavwrite(segment, outFile, afm.afReader.samplerate())
        print "...done."

def get_arbitrary_file_segments(readfilepath, segmentList):
    ''' filepath contains the input files
    segment list is a list of (start point, length), in seconds 

    returns: a list of audio samples for each segment, fs for the read file'''
    #print "reading", segmentList, "from", readfilepath

    afReader = af.AudioReader(readfilepath)
    fs = afReader.samplerate()

    sampleSegmentList = [ (x * fs, l * fs) for x, l in segmentList]

    audio_segments = [ afReader.read_frame_at_index(np.int(x), np.int(l)) for x, l in sampleSegmentList ]
    afReader.close()
    return audio_segments, fs

def write_segment_audio(writefilepath, audioSegments, fs):
    ''' given an output path, and a list of audio segments, 
    concatenate them into a single file, with blank space between each. '''
    print "Writing", len(audioSegments), "audio segments to", writefilepath

    fileDir = os.path.dirname(writefilepath)
    fileName = os.path.basename(writefilepath)
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
        
    afWriter = af.AudioWriter(writefilepath, samplerate=fs)

    for segment in audioSegments:
        afWriter.write_frame(segment)
        # Append some silence
        afWriter.write_frame( np.zeros( .5 * afWriter.samplerate()) )

    afWriter.close()
