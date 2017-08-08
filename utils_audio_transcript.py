import pandas as pd
import datetime as dt
import mido as md
import aubio as ab
import numpy as np
import librosa as lb
from matplotlib.cm import Blues
import matplotlib.pyplot as plt
import subprocess
import scipy.io.wavfile

START_DATETIME = dt.datetime(2000, 1, 1, 0, 0, 0, 0)
MIN_DELTATIME_FLOAT = 0.000001 # i.e. 1 micro-second
MIN_DELTATIME_STR = 'us'
SR = 44100
N_FFT = 4096
HOP_LENGTH = 512

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

def calc_midi_stats(filename_midi):
    """
    Compute some stats to get some sense of the content of a midi file. 
    """
    mid = md.MidiFile(filename_midi)
    
    nb_note_on = 0
    nb_note_off = 0
    for msg in mid.tracks[-1]: # only look at last track
        if msg.type == 'note_on':
            nb_note_on += 1
        if msg.type == 'note_off':
            nb_note_off += 1
            
    return nb_note_on, nb_note_off

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
    samplerate = SR // downsample
    win_s = N_FFT // downsample # fft size
    hop_s = HOP_LENGTH  // downsample # hop size
    
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

def distance_midi_cosine(x, y):
    '''
    Computes the distance between two columns of a piano-roll representation.
    (i.e. 128-dimensional binary vector)
    Use the cosine distance.
    Could be improved by adjusting weights for:
    - same note / different octave
    - harmonics
    '''
    x_null = np.sum(x) == 0
    y_null = np.sum(y) == 0
    
    if x_null and y_null:
        return(0.0)
    
    if (x_null and not y_null) or (not x_null and y_null):
        return(1.0)
         
    return( 1.0 - np.dot(x, y) / float(np.linalg.norm(x)*np.linalg.norm(y)))
        

def distance_chroma(x,y, nb_frames, nb_chromas):
    '''
    This function computes the distance between two chromagrams of the same size.
    Use L2 norm, but that can be changed.
    '''
    d = np.linalg.norm(x-y) / float(nb_frames*nb_chromas)    
#     d = nb_frames - np.sum(x*y) # No need to divide by the product of norms as the total energy has been normalised already
    
    return(d)

def compare_chomagrams(chromagram_act, chromagram_est, sr, hop_length):
    '''
    Compute the distance between an estimated chromagram (processed online) and a 
    "true" chromagram.
    Roll down the true chromagram and compute the distance for each row.
    '''
    nb_frames_est = chromagram_est.shape[0]
    nb_frames_act = chromagram_act.shape[0]    
    
    dist = []
    # Loop over the actual chromagram and compute the distance with the 
    # estimate for each row
    for k in np.arange(nb_frames_est, nb_frames_act):
        dist += [distance_chroma(chromagram_act[(k-nb_frames_est):k,], chromagram_est, nb_frames_est, chromagram_est.shape[1])]
        
    # Reshape the distance as a timeseries
    times = np.arange(0, nb_frames_act-nb_frames_est) * hop_length / float(sr)
        
    # Convert to absolute timestamps
    ts_dist = relative_ts_to_absolute_ts(times, dist)
    
    return(ts_dist)

def plot_chromagram(chromagram, sr=1.0, hop_length=1.0, ax=None, xticks_sec=True):
    '''
    Plot a chromagram as a heatmap.
    Add to an existing plot or create a new one.
    '''        
    
    if ax is None:
#         plt.figure()
        ax = plt.gca()
        
    ax.pcolor(np.transpose(chromagram), cmap=Blues, alpha=0.8)
    ax.set_frame_on(False)
    
    if xticks_sec:
        # Set the x axis to seconds
        xticks = np.linspace(0, chromagram.shape[0], 200)
        ax.set_xticks(xticks, minor=False)  
        ax.set_xticklabels(map(lambda x: "%0.1f" % x, xticks * hop_length / float(sr)), minor=False)
    
#     pcm=ax.get_children()[2]
#     plt.colorbar(pcm, ax=ax)

    return()

def plot_dtw_distance(cum_distance):
    '''
    Plot the cumulative distance as a heat map to visualise the DTW.

    Parameters
    ----------
    cum_distance : np.ndarray
        The cumulative choma/choma distance
    
    '''
    mask_rows = np.invert(np.all(np.isnan(cum_distance), 1))
    mask_cols = np.invert(np.all(np.isnan(cum_distance), 0))
    cum_distance = cum_distance[mask_rows, :][:, mask_cols]
    
    plt.pcolor(cum_distance, cmap=Blues, alpha=0.8, vmin=np.nanmin(cum_distance), vmax=np.nanmax(cum_distance))
    plt.colorbar()
    
    return()    

def synthetise(input_path, output_path, synthetise_mode = '-OwM'):
    """
    Synthetise a .midi file to a .wav file using Timidity++
    Uses the portable version fo Timidity, but we should move to normal version.
    
    E.g.
    input_path = 'C:\Users\Alexis\Business\SmartSheetMusic\Samples\Chopin_op28_3\Chopin_Op028-01_003_20100611-SMD.mid'
    output_path = 'C:\Users\Alexis\Business\SmartSheetMusic\Samples\Chopin_op28_3\output.wav'
    
    synthetise_mode = '-OwM' Generate RIFF mono format output.
    For all other options, see the "mode" options in: 
    http://www.onicos.com/staff/iz/timidity/doc/options.html    
    """
    
    timidity_folder = 'C:/Program Files/timidity'
    cmd = 'timidity {} -s 44100 {} -o {}'.format(input_path, synthetise_mode, output_path)
     
    cmd_output = subprocess.call(cmd, shell=True, cwd=timidity_folder)    
    
    return(cmd_output)

def write_wav(filename, data, rate = 44100):
    """ 
    From a numpy array, store the .wav file on the disk.
    The numpy array may be generated by Fluidsynth.
    """
    
    # Compress the data (the input format is likely to be float64)
    # Make sure that the format is readable by Librosa
    maxv = np.iinfo(np.int16).max
    lb.output.write_wav(filename, (data * maxv).astype(np.int16), rate)    
    
    return(None)

def change_file_format(filename, old_format_extension, new_format_extension, append = ''):
    """
    Remove the old extension and append the new one
    """
    filename = unmake_file_format(filename, old_format_extension)
    filename += append + new_format_extension
    
    return(filename)

def make_file_format(filename, format_extension):
    """
    Ensure that a file has the required extension.
    """
    
    if filename[len(filename)-len(format_extension):len(filename)] != format_extension:
        filename += format_extension
        
    return(filename)

def unmake_file_format(filename, format_extension):
    """
    Remove the extension of a file name.
    """
    
    if filename[len(filename)-len(format_extension):len(filename)] == format_extension:
        filename = filename[0:len(filename)-len(format_extension)]
        
    return(filename)

def calc_alignment_stats(times_cor_est, times_ori_est, times_cor_act, times_ori_act):
    """
    Compute some stats to evaluate the quality of the alignment procedure.    
    The alignment error is computed for each estimated time in the corrupted data.    
    
    Negative alignment_error --> we are late in the partition, e.g. we think we are 
    in the 4th measure, while we are actually at the 5th measure.
    Positive alignment error --> we are early in the partition. 
    
    Parameters
    ----------
    times_cor_est : np.ndarray
        The output times of the alignment procedure (timestamps in the corrupted .wav).
    times_ori_est : np.ndarray
        The output times of the alignment procedure (timestamps in the original .wav).        
    times_cor_act : np.ndarray
        The ground-truth timestamps in the corrupted .wav.
    times_cor_act : np.ndarray
        The ground-truth timestamps in the original .wav.        
    
    Returns
    -------
    alignment_error  : np.ndarray 
        The alignment error, for each times_cor_est point.
        
    """ 
    if len(times_cor_est) != len(times_ori_est) or len(times_cor_act) != len(times_ori_act): raise ValueError('Input times not matching')  
    
    # Interpolate the times_ori_act values. Extrapolation may need to be added later
#     times_ori_act_interp = np.interp(times_cor_est, times_cor_act, times_ori_act)
    times_ori_act_interp = find_alignment_times(times_cor_est, times_cor_act, times_ori_act)
    
    # Compute the alignment errors as the difference between 
    # estimate vs actual times in the original score
    alignment_error = times_ori_est - times_ori_act_interp
    
    return(alignment_error)

def find_alignment_times(times_cor_est, times_cor_act, times_ori_act):
    """
    For some input times, referring to the estimated corrupted times, return 
    the "true" times in the original data.
    
    Parameters
    ----------
    times_cor_est : np.ndarray
        The output times of the alignment procedure (timestamps in the corrupted .wav).       
    times_cor_act : np.ndarray
        The ground-truth timestamps in the corrupted .wav.
    times_cor_act : np.ndarray
        The ground-truth timestamps in the original .wav.        
    
    Returns
    -------
    times_ori_act_interp  : np.ndarray 
        The matching times in the original data.   
     
    """
    if len(times_cor_act) != len(times_ori_act): raise ValueError('Input times not matching')
    
    # Interpolate the times_ori_act values. Extrapolation may need to be added later
    times_ori_act_interp = np.interp(times_cor_est, times_cor_act, times_ori_act)
    
    return(times_ori_act_interp)

