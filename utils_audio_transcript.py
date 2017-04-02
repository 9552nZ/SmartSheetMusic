import pandas as pd
import datetime as dt
import mido as md
import aubio as ab

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
#     for i, tr in enumerate(mid.tracks):
#     for tr in mid.tracks:
    for msg in mid.tracks[1]: # only look up in the second track                     
#         if msg.type == 'note_on' and msg.velocity > 0:
#             pitches_act += [msg.note]
#             times_act += [cnt_time]
#             
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

def process_wav_file(filename_wav):
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
    
    # Estimate the pitches
    total_frames = 0
    while True:
        samples, read = s()    
        pitch = pitch_o(samples)[0]
        pitches_est += [pitch]  
        times_est += [times_est[-1] + read / float(samplerate)] if total_frames > 0 else [0]
        total_frames += read
        if read < hop_s: break     
    
    ts_est = relative_ts_to_absolute_ts(times_est, pitches_est)
    
    return(ts_est)