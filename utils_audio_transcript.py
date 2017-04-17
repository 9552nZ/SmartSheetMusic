import pandas as pd
import datetime as dt
import mido as md
import aubio as ab
import numpy as np

START_DATETIME = dt.datetime(2000, 1, 1, 0, 0, 0, 0)
MIN_DELTATIME_FLOAT = 0.000001 # i.e. 1 micro-second
MIN_DELTATIME_STR = 'us'

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
    samplerate = 44100 // downsample
    win_s = 4096 // downsample # fft size
    hop_s = 512  // downsample # hop size
    
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
