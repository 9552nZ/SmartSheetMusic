import numpy as np
from librosa.output import write_wav as lb_write_wav
from pyaudio import PyAudio, paInt16, paInt32, paFloat32
from librosa.util import frame
from sys import executable
from datetime import timedelta
from math import ceil
from os.path import getsize, splitext


# Constants
SR = 11025
N_FFT = 4096
HOP_LENGTH = 1024
WD = "C:\\Users\\Alexis\\Business\\SmartSheetMusic\\"
WD_AUDIOSET = WD + "AudioSet\\"
WD_MATCHER_EVALUATION = WD + "Samples\\TestScoreFollowing\\"
AUDIO_FORMAT_DEFAULT = "float32"
AUDIO_FORMAT_MAP = {"int16":(np.int16, paInt16), 
                    "int32":(np.int32, paInt32),
                    "float32":(np.float32, paFloat32)}
FFMPEG_PATH = '"C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"' 

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
        

def distance_chroma(x,y):
    '''
    This function computes the distance between two chromagrams of the same size.
    Use L2 norm, but that can be changed.
    '''
    d = np.linalg.norm(x-y) #/ float(nb_frames*nb_chromas)    
#     d = nb_frames - np.sum(x*y) # No need to divide by the product of norms as the total energy has been normalised already
    
    return(d)

def plot_chromagram(chromagram, sr=1.0, hop_length=1.0, ax=None, xticks_sec=True, pcol=None):
    '''
    Plot a chromagram as a heatmap.
    Add to an existing plot or create a new one.
    Can also be run online.    
    '''
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import Blues
    
    # If we run the plot online, we may pass in the output of pcolor 
    # for faster plotting. In this case, we need to turn on the interactive mode. 
    if pcol is not None:
        plt.ion()
        pcol.set_array(np.transpose(chromagram).ravel())
        pcol.autoscale()
        plt.draw()
        plt.pause(0.1)
        return(pcol)
    
    # Set an axis if none exists
    if ax is None:
        ax = plt.gca()
        
    # Draw the colormap
    pcol = ax.pcolor(np.transpose(chromagram), cmap=Blues, alpha=0.8)
    ax.set_frame_on(False)
    
    # Set the x axis to seconds 
    if xticks_sec:        
        xticks = np.linspace(0, chromagram.shape[0], 20)
        ax.set_xticks(xticks, minor=False)  
        ax.set_xticklabels(map(lambda x: "%0.1f" % x, xticks * hop_length / float(sr)), minor=False)
     
    # Add the colorbar
    pcm=ax.get_children()[0]
    plt.colorbar(pcm, ax=ax)
 
    return(pcol)

def plot_spectrogram(spectrogram, plot_data=None):
    import matplotlib.pyplot as plt        
    
    if plot_data is not None:             
        plot_data[2].set_ydata(spectrogram) # Careful about the limits
        plot_data[0].canvas.draw()
        plot_data[0].canvas.flush_events()
    else:
        plt.ion() 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot(spectrogram)
        plot_data = (fig,ax,line)
    
    return(plot_data)
        
def plot_dtw_distance(cum_distance):
    '''
    Plot the cumulative distance as a heat map to visualise the DTW.

    Parameters
    ----------
    cum_distance : np.ndarray
        The cumulative choma/choma distance
    
    '''
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import Blues
    
    mask_rows = np.invert(np.all(np.isnan(cum_distance), 1))
    mask_cols = np.invert(np.all(np.isnan(cum_distance), 0))
    cum_distance = cum_distance[mask_rows, :][:, mask_cols]
    
    plt.pcolor(cum_distance, cmap=Blues, alpha=0.8, vmin=np.nanmin(cum_distance), vmax=np.nanmax(cum_distance));
    plt.colorbar()
    
    return()

def plot_alignment(times_est_all, times_cor_act, times_ori_act, legend_items=['Estimated']):
    '''
    Create to plots to evaluate the alignment output of the matching procedure:
    - plot 1: actual corrupted time vs. actual original time / estimated corrupted times vs estimated original times
    - plot 2: alignment error, the y axis represents the difference to "true" alignment
    
    Parameters
    ----------
    times_est_all : list 
        List of tuples of np.ndarray.
        [(times_cor_est1, times_cor_act1), (times_cor_est2, times_cor_act2), ...]  
        Corresponds to the alignment output for competing alignment procedures.
        
    times_cor_act : np.ndarray
        The ground-truth corrupted times.
        
    times_ori_act : np.ndarray
        The ground-truth original times.
        
    legend_items : list
        Labels for the competing alignment procedures.
            
    '''
    
    import matplotlib.pyplot as plt
    
    # Find the largest corrupted time, to resize the x axis
    max_time = -np.inf
    for times_est in times_est_all:
        max_time = max(max_time, np.max(times_est[0]))
    
    # Plot all the alignments in the corrupted  vs. original space
    # For each point on the x axis, we can read the estimated original time in the y axis
    fig = plt.figure()
    ax = fig.add_subplot(211)    
    ax.plot(times_cor_act, times_ori_act)
    for i, times_est in enumerate(times_est_all):
        ax.plot(times_est[0], times_est[1], 'C'+str(i+1))
    plt.xlabel('Corrupted time (secs)')
    plt.ylabel('Original time (secs)')    
    plt.title('Corrupted and original times')
    plt.xlim(0, max_time)    
    plt.legend(['Actual'] + legend_items)
    
    # Plot the aligment error
    ax = fig.add_subplot(212)    
    for i, times_est in enumerate(times_est_all):
        alignment_error = calc_alignment_stats(times_est[0], times_est[1], times_cor_act, times_ori_act)        
        ax.plot(times_est[0], alignment_error, 'C'+str(i+1))
    plt.title('Alignment error')
    plt.xlabel('Corrupted time (secs)')
    plt.xlim(0, max_time)
    plt.legend(legend_items) 
    
    return()

def write_wav(filename, data, rate = 44100):
    """ 
    From a numpy array, store the .wav file on the disk.
    The numpy array may be generated by Fluidsynth.
    """
    
    # Compress the data (the input format is likely to be float64)
    # Make sure that the format is readable by Librosa
    maxv = np.iinfo(np.int16).max
    lb_write_wav(filename, (data * maxv).astype(np.int16), rate)    
    
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

def find_start_end_rests(audio_data, sr, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """
    In order to evaluate the alignment procedure, we may want to exclude the 
    (potential) start and end rests - the alignment is likely to be poor in 
    these regions, but it does not matter.
    This functions returns the estimated end of the starting rest and the
    estimated start of the ending rest.
    
    The function won't return a meaningful output if there are rests in the middle 
    of the music piece. 
    
    Parameters
    ----------
    audio_data : np.ndarray
        The raw waveform audio.       
    sr : int
        The sample rate
    
    Returns
    -------
    times_start_end_rests  : list 
        List of two items, with first being the end time of the start rest and the second item
        being the start time of the end rest, in seconds.        
    """
    
    # Compute the 3rd percentile of the envelope and  
    # deem anything below this value as silence   
    envelope = frame(audio_data, hop_length=hop_length, frame_length=n_fft).max(axis=0)
    lower_bound = np.percentile(envelope, 5.0)
    
    # Implement the search as loop, this should be faster than vectorisation
    k = 0
    while envelope[k] <= lower_bound:
        k += 1
        
    # Return 0 if there is no start rest
    if k == 0:
        time_start = 0.0
    else:
        # The first value of the output of the frame function correspond to the time of
        # n_fft, then the times are spaced according to hop_length 
        time_start = ((k-1)*hop_length + n_fft)/float(sr)
             
    j = len(envelope)-1
    while envelope[j] <= lower_bound:
        j -= 1
    
    # Return the length of the track if the is no end rest
    if j == len(envelope)-1:
        time_end = len(audio_data)/float(sr)
    else:
        time_end = ((j-1)*hop_length + n_fft)/float(sr)
            
    times_start_end_rests = [time_start, time_end]
    
    return(times_start_end_rests)

def fluidsynth_start_new_process32():
    '''
    Fluidsynth does not work with python64. We fork a 32-bit process if the 
    current interpreter is python64.
    
    Returns
    -------
    fluidsynth_start_new_process32 : bool     
    '''
    
    if '64' in executable or '36' in executable:
        fluidsynth_start_new_process32 = True
    else:
        fluidsynth_start_new_process32 = False
        
    return fluidsynth_start_new_process32        

def youtube_download_audio(yt_id, start_sec, length_sec, filename_wav_out):
    '''
    Download an audio sample from Youtube and save it as a ".wav".
    More precisely, we generate the "true" Youtube URL using youtube-dl and 
    we then use ffmpeg to download only the relevant portion.  
    Need to specify the beginning of the sample and its length in secs.
    
    An example in cmd for comparison:
    1) "C:/Program Files (x86)/ffmpeg/bin/youtube-dl.exe" -g "https://youtube.com/watch?v=TP9luRtEqjc" --quiet --extract-audio
    2) Get the ouput of the above and replace {} with it:    
    "C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe" -ss 0:00:10 -i "{}" -t 0:01:40 -acodec pcm_s16le -ac 1 -ar 16000 C:\\Users\\Alexis\\Business\\SmartSheetMusic\\Samples\\blopp.wav
    
    Parameters
    ----------
    yt_id : str
        The Youtube id, that is, anything right to "v=". E.g. "TP9luRtEqjc"
        
    start_sec : float > 0
        The starting second for the sample
        
    length_sec : float > 0
        The length we want to download
        
    filename_wav_out : str
        The address to save the ".wav" file
        
    Returns
    -------
    out : bool
        False if the youtube-dl command has failed. True otherwise.
    '''
    
    from subprocess import check_output, call
    
    # Sample rate for the output ".wav" file
    sr = SR
    
    # Reformat the start_sec and length_sec 
    start_sec = str(timedelta(seconds=start_sec))
    length_sec = str(timedelta(seconds=length_sec))
    
    # The target path for the youtube-dl 
    ytdl_exe_path = '"C:/Program Files (x86)/ffmpeg/bin/youtube-dl.exe"'
    
    # The root URL for Youtube
    yt_url = "https://youtube.com/watch?v={}".format(yt_id)

    # Build the Youtube command
    ytdl_command = "{} -g {} --quiet --extract-audio".format(ytdl_exe_path, yt_url)
    
    # Need to execute the youtube-dl in a try block as the command fails 
    # if the video has been deleted.
    try:
        real_url = check_output(ytdl_command)
        real_url = real_url[0:len(real_url)-1] # remove the return character
        
        # The ffmpeg (may need to change the codec here) 
        ffmpeg_command = '{} -ss {} -i "{}" -t {} -acodec pcm_s16le -ac 1 -ar {} {}'.format(FFMPEG_PATH, start_sec, real_url, length_sec, sr, filename_wav_out)
    
        call(ffmpeg_command)
        out = True
        
    except:
        out = False
        
    return(out)

def convert_audio_file(filename_in, filename_wav_out, sr):
    '''
    Convert an audio file to another audio file using ffmpeg.
    We may use the function to change the format (e.g. mp3 to wav), the codec,
    the sample rate, the number of chanels...
    
    Parameters
    ----------
    filename_in : str
        The path for the input file, e.g. a wav or mp3 file
        
    filename_wav_out : str
        The path for the output file.
        
    sr : int > 0
        The sample rate.
    '''
        
    from subprocess import call
     
    ffmpeg_command = '{} -i "{}" -acodec pcm_s16le -ac 1 -ar {} {}'.format(FFMPEG_PATH, filename_in, sr, filename_wav_out)
    call(ffmpeg_command)
    
    return

def calc_nb_segment_stft(hop_length, nb_sample):
    '''
    Calculate the number of expected segments for a librosa STFT transform.    
    '''
    return(nb_sample/hop_length+1)
    
def calc_nb_sample_stft(sr, hop_length, nb_sec):
    '''
    Calculate how many samples we need so as the librosa STFT matches 
    exactly the nb_sec input.
    '''
    return(int(ceil(sr*float(nb_sec)/hop_length)) * hop_length-1)
    

def record(record_sec, sr=SR, audio_format="int16", save=False, filename_wav_out="file.wav"):
    '''
    Record the input sound from the micro using pyaudio.
    We can save the output into a .wav file or return as numpy array.
    
    Parameters
    ----------
    record_sec : float > 0
        The number of seconds we want to record
        
    sr : int 
        The sampling rate
        
    save : bool 
        Set to true to save the output
        
    filename_wav_out : str
        The address to save the ".wav" file
        
    Returns
    -------
    audio_data : np.ndarray
        The recorded waveform as a numpy array.
        
    '''     
    channels = 1
    chunk = 2048
     
    audio = PyAudio()
     
    # Start Recording
    stream = audio.open(format=AUDIO_FORMAT_MAP[audio_format][1], 
                        channels=channels,
                        rate=sr, 
                        input=True,
                        frames_per_buffer=chunk)
    
    print("Recording...")
    frames = []
     
    for _ in range(0, int(sr / float(chunk) * record_sec)):
        data = stream.read(chunk)
        frames.append(data)        
    
    print("Finished recording.")
          
    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Convert the pyaudio string to numpy number format
    audio_data = np.hstack(map(lambda x: np.fromstring(x,dtype=AUDIO_FORMAT_MAP[audio_format][0]), frames))
    
    # Reshape the data to output the desired number of channels
    if channels == 2:  
        audio_data = np.reshape(audio_data, (len(audio_data)/channels, channels))         
    
    if save:
        lb_write_wav(filename_wav_out, audio_data, sr, norm=False)
    
    return(audio_data)

def start_and_record(filename, filename_new, sr=SR):
    '''
    Launch a wave file via MPC (branching a new process) and record it through the mic.
    
    Parameters
    ----------
    filename : str
        The full path for the wave file that we want to launch and record.
        
    filename_new : str
        The full path for the new wave file that has been recorded.
        
    sr : int 
        The sampling rate
             
    '''
    
    from subprocess import Popen
    
    # Flag to detach teh process from the python thread
    # Without it, we would wait for the output of the command.
    DETACHED_PROCESS = 0x00000008    
    
    # Find the lenghts (in secs) of the target wave file.
    record_length = get_length_audio_file(filename)    
    
    # Launch the wav via MPC.
    cmd = r'C:\\Program Files\\MPC-HC\\mpc-hc64.exe {}'.format(filename)        
    p = Popen(cmd,shell=False,stdin=None,stdout=None,stderr=None,close_fds=True,creationflags=DETACHED_PROCESS)
    
    # Record. 
    # Add 0.1 sec as the starting of the recording takes some time
    # Store in int16 format (in float32, the recorded wav appears corrupted for some reason).
    record(record_length + 0.1,   
           sr=sr, 
           audio_format="int16", 
           save=True, 
           filename_wav_out=filename_new)            

def get_length_audio_file(filename):
    '''
    Get the length in seconds of an audio file (wave or mp3).
    
    Parameters
    ----------
    filename : str
        The full path for the wave file.
        
    Returns:
    s : float
        The length of the file in seconds.
    
    '''
    
    _, extension = splitext(filename)
    
    if extension in ['.wav', '.wave']:
        
        # (in python2, f = open(filename,"r"))
        f = open(filename, encoding='Latin-1')
    
        #read the ByteRate field from file (see the Microsoft RIFF WAVE file format)
        #https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
        #ByteRate is located at the first 28th byte
        f.seek(28)
        a=f.read(4)
        
        #convert string a into integer/longint value
        #a is little endian, so proper conversion is required
        byte_rate = 0
        for i in range(4):
            byte_rate = byte_rate + ord(a[i])*pow(256,i)
        
        #get the file size in bytes
        file_size = getsize(filename)
        f.close()  
        
        #the duration of the data, in seconds, is given by
        s = ((file_size-44))/float(byte_rate)
        
    elif extension == '.mp3':   
        from mutagen.mp3 import MP3
        audio = MP3(filename)
        s = audio.info.length
        
    else:
        raise TypeError("Need either wave or mp3 file")
    
    return(s)

def calc_mean_random_distance(chromagram):
    '''
    Take a chromagram as input and calculate the mean distance between 
    two random chromas.
    This may serve as a comparison to see how the DTW alignment performs.

    Parameters
    ----------
    chromagram : np.ndarray (nb frames x nb chromas)
        The input chromagram. We assume it has been adequately normalised
               
    Returns
    -------
    mean_dist : float
        The mean random distance.    
    
    '''
    
    len_chromagram = chromagram.shape[0]
    nb_boot = 1000
    boot_idx = np.random.randint(0, len_chromagram, size=(nb_boot,2))
    
    dists = np.empty(nb_boot)
    for k in range(nb_boot):
        dists[k] = distance_chroma(chromagram[boot_idx[k,0],:], chromagram[boot_idx[k,1],:])
        
    mean_dist = np.mean(dists)
    
    return(mean_dist)

def find_relative_mins(arr, order, nb_relative_mins):
    '''
    Find the local / relative minima in a 2D array (find the per-column mins).
    
    Parameters
    ----------
    arr : np.ndarray (n x m)
        The array for which we find the local mins (find the per-column mins).
        
    order : int
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
        
    nb_relative_mins : int
        How many local minima we want to return. 
        We select the nb_relative_mins of the minima.
        
    Returns
    -------    
    relative_mins_idxs : np.ndarray (nb_relative_mins x m)
        The indices of the local minima. If there were not enough minima, we return NaNs.

    relative_mins_values : np.ndarray (nb_relative_mins x m)
        The values of the local minima. If there were not enough minima, we return NaNs.
            
    '''
    
    # Import the local min function from Scipy here as we won't 
    # use it anywhere else
    from scipy.signal import argrelmin
    
    arr_copy = arr.copy()
    
    # Get the shape and replace the NaNs with +inf
    nb_col = arr_copy.shape[1]
    arr_copy[np.isnan(arr_copy)] = np.inf
    
    # Set up placeholders for the output
    relative_mins_idxs = np.ones((nb_relative_mins, nb_col))*np.nan
    relative_mins_values = np.ones((nb_relative_mins, nb_col))*np.nan
    
    # Loop over columns
    for k in range(nb_col):
        
        # Get the indices of the local mins (all of them at a given order)        
        idxs = argrelmin(arr_copy[:, k], order=order, mode='wrap')[0]
        
        # Sort the local mins
        values = arr_copy[idxs, k]
        idxs_to_sort = np.argsort(values)
        idxs_sorted = idxs[idxs_to_sort]
        values_sorted = values[idxs_to_sort]
        
        # Only return the smallest mins 
        nb_idxs = len(idxs)        
        mask = np.arange(0, min(nb_idxs, nb_relative_mins))
        relative_mins_idxs[mask, k] = idxs_sorted[mask]
        relative_mins_values[mask, k] = values_sorted[mask]
        
    return(relative_mins_idxs, relative_mins_values)

def dtw(C, weights_mul=np.array([1.0, 1.0, 1.0]), subseq=False):
                
    D = C.copy()

    # Set starting point to C[0, 0]    
    D[0, 0:] = np.cumsum(C[0,:])    
    
    if subseq:
        D[0:, 0] = C[:, 0]
    else:
        D[0:, 0] = np.cumsum(C[:, 0])        
    
    r, c = np.array(C.shape)-1    
    for k in range(1, r+c):
        # We have i>=0, i<r, j>0, j<c and j-i+1=k
        i = np.arange(max(0, k-c), min(r, k))
        j = i[::-1] + k - min(r, k) - max(0, k-c)
        
        D_tmp = np.array([D[i, j] + D[i+1, j+1] * weights_mul[0],
                          D[i, j+1] + D[i+1, j+1] * weights_mul[1],         
                          D[i+1, j] + D[i+1, j+1] * weights_mul[2]])
        
        D[i+1, j+1] = D_tmp[np.argmin(D_tmp, axis=0), np.arange(0, len(i))]  
        
    return(D)

def dtw2(C, weights_mul=np.array([1.0, 1.0, 1.0]), subseq=False):
                 
    D = C.copy()
    D_steps = np.ones(C.shape, dtype='int32')
 
    # Set starting point to C[0, 0]    
    D[0, 0:] = np.cumsum(C[0,:])
    D_steps[0, 0:] = np.cumsum(D_steps[0,:])
     
    if subseq:
        D[0:, 0] = C[:, 0]
    else:
        D[0:, 0] = np.cumsum(C[:, 0])
        D_steps[0:, 0] = np.cumsum(D_steps[:, 0])
     
    r, c = np.array(C.shape)-1    
    for k in range(1, r+c):
        # We have i>=0, i<r, j>0, j<c and j-i+1=k
        i = np.arange(max(0, k-c), min(r, k))
        j = i[::-1] + k - min(r, k) - max(0, k-c)
         
        D_tmp = np.array([D[i, j] + D[i+1, j+1] * weights_mul[0],
                          D[i, j+1] + D[i+1, j+1] * weights_mul[1],         
                          D[i+1, j] + D[i+1, j+1] * weights_mul[2]])
         
        D_steps_tmp = np.array([D_steps[i, j], D_steps[i, j+1], D_steps[i+1, j]]) 
         
        argmins = np.argmin(D_tmp, axis=0)
        D_steps[i+1, j+1] += D_steps_tmp[argmins, np.arange(0, len(argmins))]
        D[i+1, j+1] = D_tmp[argmins, np.arange(0, len(argmins))]  
         
    return(D, D_steps)

def figure():
    '''
    Wrapper around plt.figure() to create a figure and position it.
    '''
    import matplotlib.pyplot as plt
    f = plt.figure();
    f.canvas.manager.window.wm_geometry('1432x880+2366+35')