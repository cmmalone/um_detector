
def signalFrames( raw ):
    s_markers, b_markers = signalBackgroundMarkers( raw )
    s_frames = []
    s_values = []
    frames = raw.readframes(raw.getnframes())
    for block in s_markers:
        start = block[0]
        end =   block[1]
        start_frame = raw.getmark(start)[1]
        end_frame   = raw.getmark(end)[1]
#        for frame in range(start_frame, end_frame):
        ### black magic
        ### https://robotwhale.wordpress.com/tag/scikit-learn/
        raw.setpos( start_frame )
        str_frames = raw.readframes( end_frame - start_frame )
        time_data = np.fromstring(str_frames, np.short).byteswap() 
        s_frames.append(time_data)
    return s_frames, s_markers




def upsampleNorm( frames ):
    lengths = [len(ii) for ii in frames]
    longest = max( lengths )
    for block in frames:
        print type(block)
        block = list(block)
        while len(block)<longest-1:
            interp_point = random.randrange( 0, len(block)-2 )
            ii_val = block[interp_point]
            jj_val = block[interp_point+1]
            new = (ii_val+jj_val)/2
            block.insert(interp_point+1, new)
        block = np.asarray(block)
    return frames


