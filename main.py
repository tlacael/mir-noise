'''
@author: Christopher Jacoby, Tlacael Esparza
MIR Fall 2012: Prof. Juan Bello

Final Project - Feature Extraction & Clustering of Noise Data
'''

import argparse
import mirlib.audiofile as af

def getfeatures(args):
    filepath = args.audiofile
    audio_seg = args.audio_seg_length
    print "Feature Extraction Mode"

    afm = af.audiofile_manager(filepath, audio_seg)
    # For each chunk of audio
    while afm.HasMoreData():
        audioChunk, index = afm.GetNextSegment()
        # Only look at Left if there's more than that
        if (audioChunk.shape) > 1 and audioChunk.shape[1] > 1:
            audioChunk = audioChunk[:,0]

        print "Read %d sample chunk of audio" % (len(audioChunk))
    
        # 1. Get Onsets

        # 2. Get Segments from those offsets

        # 3. Get the MFCCs for each segment / event

        # 4. Time-average for each segment / event

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
