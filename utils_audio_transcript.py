import pandas as pd
import datetime as dt
import mido as md
import aubio as ab
import numpy as np
import librosa as lb
from matplotlib.cm import Blues

START_DATETIME = dt.datetime(2000, 1, 1, 0, 0, 0, 0)
MIN_DELTATIME_FLOAT = 0.000001 # i.e. 1 micro-second
MIN_DELTATIME_STR = 'us'
SAMPLERATE = 44100
WIN_S = 4096
HOP_S = 512

def to_clipboard(arr):
    df = pd.DataFrame(arr)
    df.to_clipboard()
    print "Data loaded in clipboard."
    return()

def relative_ts_to_absolute_ts(times_sec, data_in):    
    """
    Transform the pitch array into time series
    """
    start = dt.datetime(year=2000,month=1,day=1)
    datetimes = map(lambda x:dt.timedelta(seconds=x)+start,times_sec)
    ts = pd.Series(data_in,index=datetimes)
    
    return ts


def process_midi_file(filename_midi):
    """
    Read the .midi file, extract the notes and reshape it
    """
    
    mid = md.MidiFile(filename_midi)
    
    # First find the tempo and make sure there is no tempo change
    # Warning: This will break if the tempos are not sorted in the midi file 
    tempo_values = []
    tempo_ticks = [] 
    for tr in mid.tracks:
        for msg in tr:    
            if msg.type == 'set_tempo':
                tempo_values += [msg.tempo]
                tempo_ticks += [msg.time]
    if len(tempo_values) == 0 : raise ValueError('The tempo is not set properly')
    
    if len(mid.tracks) != 2 : raise ValueError('The midi file does not have two tracks, likely to get errors')
                
    # Now retrieve the actual pitches
    pitches_act = []
    times_act = []
    cnt_time = 0
    cnt_tick = 0  
    for msg in mid.tracks[1]: # only look up in the second track                     
        # Find the current tempo
        tempo_ticks_tmp = [x for x in tempo_ticks if x <= cnt_tick]
        tempo = tempo_values[len(tempo_ticks_tmp)-1]
        cnt_tick += msg.time
        cnt_time += md.tick2second(msg.time, mid.ticks_per_beat, tempo)# 714285
        
        if msg.type == 'note_on':
            pitches_act += [msg.note if msg.velocity > 0 else 0]
            times_act += [cnt_time]                     
            
    ts_act = relative_ts_to_absolute_ts(times_act, pitches_act)
    
    # Remove duplicate indices and fill forward the pitches (to avoid linear interpolation) 
    ts_act_clean = ts_act[~ts_act.index.duplicated(True)]
    ts_act_clean = ts_act_clean.add(ts_act_clean.shift(1).shift(-1, freq='0.001ms'), fill_value=0)
            
    return(ts_act_clean)


def process_wav_file(filename_wav, start_sec = 0.0, end_sec = float("inf")):
    """
    Read the .wav file, process it and estimate the Midi pitch number using aubio
    """
    downsample = 1
    samplerate = SAMPLERATE // downsample
    win_s = WIN_S // downsample # fft size
    hop_s = HOP_S  // downsample # hop size
    
    # Read the .wav file
    s = ab.source(filename_wav, samplerate, hop_s)
    samplerate = s.samplerate
    
    tolerance = 0.8
    
    # Set up the pitch-estimation object
    pitch_o = ab.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    
    pitches_est = []
    times_est = []
    
    # Set the cursor to the first desired frame
    first_frame = int(samplerate * start_sec)
    s.seek(first_frame)
        
    # Estimate the pitches
    total_frames = 0 # can be removed
    while True:
        samples, read = s()    
        pitch = pitch_o(samples)[0]
        # Add 2x the pitch as we have two timestamps, one for the beginning and
        # one for the end of the frame
        pitches_est += [pitch, pitch]
        times_est += [times_est[-1] + MIN_DELTATIME_FLOAT,  times_est[-1] + MIN_DELTATIME_FLOAT + read / float(samplerate)] if total_frames > 0 else [0.0, read / float(samplerate) - MIN_DELTATIME_FLOAT]
        total_frames += read
        if read < hop_s or times_est[-1] >= end_sec-start_sec: break     
    
    # We want the timestamp to represent the beginning of the frame
    # Up to here, they represent where the hop is happening 
    # Hence, we need to take out (win_s-hop_s) / float(samplerate) 
    times_est = map(lambda x: x - (win_s-hop_s) / float(samplerate),times_est)
    
    # Convert to absolute timestamps
    ts_est = relative_ts_to_absolute_ts(times_est, pitches_est)
    
    # Get rid of the negative timestamps (has the effect of removing the 
    # estimated pitches for which we do not have a full win-s)
    ts_est = ts_est[ts_est.index >= START_DATETIME]
    
    return(ts_est)

def distance_midi_pitch(x, y):
    """
    Compute the distance between two arrays of midi numbers.
    The function assumes that the midi data has been discretised.
    The distance is defined as:
    - 0 is the values are the same
    - 1 if the abs difference is <=3
    - 2 otherwise     
    """
    nb_tot = len(x)
    
    d = abs(x - y)
        
    idx_non_null = (d != 0)
    nb_null = nb_tot - np.sum(idx_non_null)
    nb_small = np.sum(d[idx_non_null] <= 3)
    
    # Compute the different contributions    
    dist_null = 0. * nb_null 
    dist_small = 1. * nb_small
    dist_large = 2 * (nb_tot - nb_null - nb_small)          
    
    # Find the average distance (normalise per number of points)
    d_mean = (dist_null + dist_small + dist_large) / float(nb_tot)
    
    return(d_mean)

def spectrogram_to_chromagram(spectrogram, frequencies):
    '''
    This function takes a spectrogram as input and their corresponding frequencies
    and map them to a chromagram.
    The reference frequency for the used scale in C3.
    
    Reference paper:
    http://jim.afim-asso.org/jim12/pdf/jim2012_08_p_osmalskyj.pdf
    
    The frequencies taken as input may need to be checked (not entirely sure of
    what aubio.pvoc outputs
    '''
    
    nb_chroma = 12
    shape_spectrogram = np.shape(spectrogram)
    
    # Reference frequency, i.e. C3
    # Shall we use A4 as ref freq (440Hz) ?
    f_ref = 130.80      
    
    # Map the input frequencies to chroma numbers
    # The first item is NaN as the first frequency is 0Hz 
    # (to be confirmed when checking the output of pvoc)
    chroma_bands = np.concatenate(([np.NAN], np.round(nb_chroma*np.log2((frequencies[1:,])/f_ref)) % nb_chroma))
    
    # Loop over the chroma numbers and sum over the amplitudes in the 
    # associated buckets
    chroma_raw = np.zeros([shape_spectrogram[0], nb_chroma])
    for c in np.arange(nb_chroma):
        mask = chroma_bands == float(c)
        chroma_raw[:, c] = np.sum(spectrogram[:, mask], 1) # Need to take sum of square?
        
    # Rescale the chromagram such that the total energy is 1
    # for each time frame
    chromagram = chroma_raw / np.sum(chroma_raw,1)[:,None]
       
    return(chromagram)

def get_wav_spectrogram(filename, samplerate = SAMPLERATE, win_s = WIN_S, hop_s = HOP_S, start_sec = 0.0, end_sec = 100000.0, read_lib = 'librosa'):#float("inf")
    '''
    This function extract the STFT for a .wav file using the aubio.pvoc function.
    The spectrogram output is associated with a set of frequencies (but these would need to 
    be checked).
    
    We can also extract only part of the .wav file by setting start_sec and end_sec
    '''
         
    fft_s = win_s // 2 + 1 # fft size
    
    # Phase vocoder
    # The phase vocoder also does the windowing (HanningZ)
    # c.f. https://github.com/aubio/aubio/blob/master/src/spectral/phasevoc.h
    pv = ab.pvoc(win_s, hop_s)
    
    # Array to store spectrogram
    spectrogram = np.zeros([0, fft_s], dtype=ab.float_type)
    
    if read_lib == 'aubio':
        # Read the .wav file    
        s = ab.source(filename, 0, hop_s)#samplerate    
        samplerate = s.samplerate
    
        # Set the cursor to the first desired frame
        first_frame = int(samplerate * start_sec)
        s.seek(first_frame)    
        
        time_tot = 0
        while True:    
            samples, read = s()                           
            spectrogram = np.vstack((spectrogram,pv(samples).norm)) 
            time_tot += read / float(samplerate)        
            if read < hop_s or time_tot >= end_sec-start_sec: break
                
    elif read_lib == 'librosa':
        # Read the .wav file
        print "Loading {} at {}Hz       \r".format(filename, samplerate)
        s = lb.core.load(filename, sr = samplerate, offset = start_sec, duration = end_sec-start_sec)
        print "Loading completed \r"
                
        nb_iter = s[0].shape[0] // hop_s # We miss the last few frames
        for i in range(nb_iter):
            samples = s[0][i*hop_s:(i+1)*hop_s]
            spectrogram = np.vstack((spectrogram,pv(samples).norm))           
                    
    frequencies = (samplerate / 2.) / float(fft_s-1) * np.arange(fft_s)        
    
    # Return results as a dict
    return(dict(spectrogram=spectrogram, frequencies=frequencies, samplerate=samplerate, win_s=win_s, hop_s=hop_s))

def get_audio_data_spectrogram(audio_data, samplerate = SAMPLERATE, win_s = WIN_S, hop_s = HOP_S):
    
    fft_s = win_s // 2 + 1 # fft size
    
    # Phase vocoder
    pv = ab.pvoc(win_s, hop_s)
    
    # Array to store spectrogram
    spectrogram = np.zeros([0, fft_s], dtype=ab.float_type)
    
    len_audio_data = len(audio_data)
    cnt = 0    
    while True:
        sample = np.array(audio_data[cnt*hop_s:(cnt+1)*hop_s], dtype=ab.float_type)
        spectrogram = np.vstack((spectrogram,pv(sample).norm))        
        if len_audio_data < (cnt+2)*hop_s: break
        cnt += 1
        
    frequencies = (samplerate / 2.) / float(fft_s-1) * np.arange(fft_s) 
        
    return(dict(spectrogram=spectrogram, frequencies=frequencies))

def get_filename_spectrogram(filename, samplerate, win_s, hop_s):
    
    filename_spectrogram = filename + "_spectrogram_S{}_W{}_H{}.npy".format(samplerate, win_s, hop_s)
    
    return(filename_spectrogram)

def write_spectrogram_to_disk(wd, filename, samplerate, win_s, hop_s):
    
    filename_wav = wd + filename + ".wav"
    filename_spectrogram = get_filename_spectrogram(wd + filename, samplerate, win_s, hop_s)
    spec_data_act = get_wav_spectrogram(filename_wav, samplerate = samplerate, win_s = win_s, hop_s = hop_s)
    np.save(filename_spectrogram, spec_data_act)
    
def load_spectrogram_from_disk(wd, filename, samplerate, win_s, hop_s):

    filename_spectrogram = wd + filename + "_spectrogram_S{}_W{}_H{}.npy".format(samplerate, win_s, hop_s)
    spec_data_act = np.load(filename_spectrogram).item()
    
    return(spec_data_act)     


def distance_chroma(x,y, nb_elmts):
    '''
    This function computes the distance between two chromagrams of the same size.
    Use L2 norm, but that can be changed.
    '''
    d = np.linalg.norm(x-y) / float(nb_elmts)
    
    return(d)

def compare_chomagrams(chromagram_act, chromagram_est, samplerate, hop_s):
    '''
    Compute the distance between an estimated chromagram (processed online) and a 
    "true" chromagram.
    Roll down the true chromagram and compute the distance for each row.
    '''
    nb_frames_est = chromagram_est.shape[0]
    nb_frames_act = chromagram_act.shape[0]    
    
    # Normalise the distance by the number of elements in the window
    nb_elmts = nb_frames_est * chromagram_est.shape[1]

    dist = []
    # Loop over the actual chromagram and compute the distance with the 
    # estimate for each row
    for k in np.arange(nb_frames_est, nb_frames_act):
        dist += [distance_chroma(chromagram_act[(k-nb_frames_est):k,], chromagram_est, nb_elmts)]
        
    # Reshape the distance as a timeseries
    times = np.arange(0, nb_frames_act-nb_frames_est) * hop_s / float(samplerate)
        
    # Convert to absolute timestamps
    ts_dist = relative_ts_to_absolute_ts(times, dist)
    
    return(ts_dist)

def plot_chromagram(chromagram, samplerate, hop_s, ax):
    '''
    Plot a chromagram as a heatmap
    '''    

    ax.pcolor(np.transpose(chromagram), cmap=Blues, alpha=0.8)
    ax.set_frame_on(False)
    ax.set_xticks(np.arange(chromagram.shape[0]), minor=False)    
    ax.set_xticklabels(map(lambda x: "%0.1f" % x, np.arange(chromagram.shape[0]) * hop_s / float(samplerate)), minor=False)

    return()