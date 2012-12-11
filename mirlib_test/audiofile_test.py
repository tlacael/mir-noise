import argparse
import mirlib.audiofile as af

def segment_files(filepath, seg_length):
    af.segment_audio_files(filepath, segment_length)

def get_segments(filepath, seg_length):
    afm = af.audiofile_manager(filepath, seg_length)
    while afm.HasMoreData():
        segment, index = afm.GetNextSegment()

        print "%d: %d | (%d of %d)" % (index, segment.shape[0], index * afm.seg_length_samps, afm.afReader.numsamples())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Audio file for processing.")
    parser.add_argument("-l", "--segment_length", help="Length of audio segment (seconds)",
                        default=(30), type=int)
    parser.add_argument("--segment_files", help="Convert input file into smaller segments of audio data",
                        action="store_true")

    args = parser.parse_args()
    filepath = args.audio_file
    segment_length = args.segment_length

    if args.segment_files:
        segment_files(filepath, segment_length)
    else:
        get_segments(filepath, segment_length)
    
if __name__ == "__main__":
    main()
