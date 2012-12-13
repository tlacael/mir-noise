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
    def __init__(self, filepath, segment_length):
        self.filepath = filepath
        self.filename = os.path.split(filepath)[1]
        self.index = 0

        self.afReader = af.AudioReader(filepath)
        self.seg_length_samps = int(np.round(segment_length * self.afReader.samplerate()))

    def HasMoreData(self):
        return (self.index * self.seg_length_samps) < self.afReader.numsamples()

    def GetNextSegment(self, bMono=True):
        segment = self.afReader.read_frame_at_index(self.index * self.seg_length_samps, self.seg_length_samps)
        self.index += 1

        if bMono and (segment.shape) > 1 and segment.shape[1] > 1:
            segment = segment[:, 0]
        
        return segment, (self.index - 1)


def segment_audio_files(filepath, segment_length, result_dir=RESULT_DIR):
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
