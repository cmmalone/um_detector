#!/usr/bin/python
import random
import numpy as np

def signalBackgroundMarkers(raw):
    markers    = raw.getmarkers()
    signal_markers = []
    background_markers = []
    for mm in range(0, len(markers)-1):
        if mm%2 == 0:
            signal_markers.append( (markers[mm], markers[mm+1]) )
        else:
            background_markers.append( (markers[mm], markers[mm+1]) )
    background_markers = background_markers[:-1]

    return signal_markers, background_markers





def findWindowLength( block ):
    start = block[0][1]
    end   = block[1][1]
    return end-start


def findMaxWindow( markers ):
    max_window = 0
    for block in markers:
        print block
        if findWindowLength(block)>max_window:
            max_window = findWindowLength(block)

    return max_window



def normedFrames( raw_frames, window_length, start_points=None, n_samples=100 ):
    """ return frames, either signal or background, where each
        data point has the same number of frames (specified by window_length)
        if it's making a signal training set, start_points should
        be the start frames of the signal samples
        otherwise, will randomly sample, with n_samples points returned 
    """
    nframes = raw_frames.getnframes()
    frames = []
    if start_points:
        for block in start_points:
            start = block[0]
            print start
            raw_frames.setpos( start[1] )
            str_frames = raw_frames.readframes( window_length )
            time_data = np.fromstring(str_frames, np.short).byteswap()
            frames.append(time_data.astype(float) )
    else:
        for ii in range(0, n_samples):
            rand_start = random.randrange( 0, nframes-window_length ) 
            raw_frames.setpos( rand_start )
            str_frames = raw_frames.readframes( window_length )
            time_data = np.fromstring(str_frames, np.short).byteswap()
            frames.append(time_data.astype(float))
    return frames


