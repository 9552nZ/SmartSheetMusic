import utils_audio_transcript as utils
import aubio as ab
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


start_datetime = dt.datetime(2000, 1, 1, 0, 0, 0, 0)

def distance_midi_pitch(x, y):
    
    def distance(x, y):        
        x = int(round(x))
        y = int(round(y))
        diff_xy = abs(x-y)
        if diff_xy == 0:
            d = 0
        elif diff_xy <= 3:
            d = 1
        else: 
            d = 2                            
        
        return(d)
    
    # Preprocess by removing the nans
    idx_clean = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))) 
    d = map(distance, x[idx_clean], y[idx_clean])    
    d_tot = sum(d) / float(sum(idx_clean))
    
    return(d_tot)

def compare_act_and_est_snippet(ts_act, ts_est):
    """
    Given the time series of the actual midi file and an estimate of the pitches
    estimated with Aubio, find where we are likely to be in the midi file 
    """
    
    # Length of the estimated snippet as a timedelta
    delta_est = ts_est.index[-1].to_datetime()- start_datetime
    
    # Size of the window used for comparing the timeseries 
    win_s = delta_est
    hop_s = 0.10 # increment in seconds
    start_window = start_datetime + dt.timedelta(0.0, 0.0, 0.0) 
    end_window = start_datetime + (start_window - start_datetime) + win_s
    
    # Gather the distance data
    values_dist = []
    timestamps_dist = []
    
    # Loop over the actual midi file, jumping hop_s seconds per iteration
    while end_window <= ts_act.index[-1]:
        # Filter the actual timeseries on relevant times
        # Unfortunately, we exclude the first and last timestamps 
        idxs = np.logical_and(ts_act.index >= start_window, ts_act.index <= end_window)
        timestamps = ts_act.index[idxs]        
        values_act_win = ts_act.values[idxs] 
        
        # Add back the first and last timestamps in case we have excluded them
        if timestamps[0] != start_window:
            timestamps = timestamps.insert(0, start_window)
            values_act_win = np.hstack(([np.NAN], values_act_win))
        if timestamps[-1] != end_window:
            timestamps = timestamps.insert(len(timestamps), end_window)
            values_act_win = np.hstack((values_act_win, [np.NAN]))
        
        # Now, shift the actual timeseries so as the first and last timestamps 
        # match with the estimated timeseries 
        ts_act_win = pd.Series(values_act_win, timestamps - (start_window - start_datetime))
                    
        # Take the union of all available timestamps
        timestamps_all = np.union1d(ts_act_win.index, ts_est.index)
        
        # Fit both timeseries to the timestamps union
        ts_act_win = ts_act_win.reindex(timestamps_all, method='pad')
        ts_est_win = ts_est.reindex(timestamps_all, method='pad')
        
        # Now, compute the distance 
        dist = distance_midi_pitch(ts_act_win.values, ts_est_win.values)
        
        timestamps_dist.append(end_window)
        values_dist.append(dist) 
        
        # Move forward by hop_s seconds
        start_window += dt.timedelta(0, hop_s, 0) 
        end_window += dt.timedelta(0, hop_s, 0)
    
    # Reshape the data as a timeseries and interpolate flat (for plotting)
    ts_dist = pd.Series(values_dist, timestamps_dist) 
    ts_dist = ts_dist.add(ts_dist.shift(1).shift(-1, freq='0.001ms'), fill_value=0)
        
    return(ts_dist)

# Input the files names used for the test
wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Badinerie/"
filename = "badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"
filename_midi = wd + filename + ".mid"

ts_act = utils.process_midi_file(filename_midi)
# ts_est = utils.process_wav_file(filename_wav)

ts_est = utils.process_wav_file(filename_wav, start_sec = 48.0, end_sec = 51.0)

start = timer() 
ts_dist = compare_act_and_est_snippet(ts_act, ts_est)
end = timer()
print(end - start)

midi_range = (65.0, 90.0)
plt.subplot(311)
ts_act.plot()
plt.legend(['Midi File Pitch'])
axis1 = plt.axis()
plt.axis([axis1[0], axis1[1], midi_range[0], midi_range[1]])
plt.subplot(312)
ts_est.plot()
plt.axis([axis1[0], axis1[1], midi_range[0], midi_range[1]])
plt.legend(['Estimated Pitch'])
plt.subplot(313)
ts_dist.plot()
plt.legend(['Distance'])
plt.show()
