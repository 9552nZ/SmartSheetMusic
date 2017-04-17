import utils_audio_transcript as utils
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
    
def compare_act_and_est_snippet(ts_act, ts_est):
    """
    Given the time series of the actual midi file and an estimate of the pitches
    estimated with Aubio, find where we are likely to be in the midi file 
    
    First version of the function, it only computes the distance between the true 
    values and the estimate for the union of timestamps (no resampling)
    
    Has been optimised to minimise running time  
    """
    
    # Start by replacing the NaNs with zeros
    # Round the values to the nearest integer
    ts_act = ts_act.replace(np.NaN, 0.0).apply(round)
    ts_est = ts_est.replace(np.NaN, 0.0).apply(round)
    
    # Length of the estimated snippet as a timedelta
    delta_est = ts_est.index[-1].to_datetime()- utils.START_DATETIME
    
    # Size of the window used for comparing the timeseries 
    win_s = delta_est
    hop_s = 0.10 # increment in seconds
    start_window = utils.START_DATETIME + dt.timedelta(0.0, 0.0, 0.0) 
    end_window = utils.START_DATETIME + (start_window - utils.START_DATETIME) + win_s
    
    # Gather the distance data
    values_dist = []
    timestamps_dist = []
        
    # Loop over the actual midi file, jumping hop_s seconds per iteration
    cnt_start = 0
    cnt_end = 0
    while end_window <= ts_act.index[-1]:
        
        # Filter the actual timeseries on relevant times
        # Unfortunately, we exclude the first and last timestamps 
        while ts_act.index[cnt_start] < start_window:
            cnt_start += 1
    
        while ts_act.index[cnt_end] < end_window:
            cnt_end += 1
        
        timestamps = np.array(ts_act.index[cnt_start:cnt_end])
        values_act_win = ts_act.values[cnt_start:cnt_end]         
        
        # Add back the first and last timestamps in case we have excluded them
        if timestamps[0] != np.datetime64(start_window):
            timestamps = np.hstack(([np.datetime64(start_window)], timestamps))
            values_act_win = np.hstack(([ts_act.values[cnt_start-1]], values_act_win))
        if timestamps[-1] != np.datetime64(end_window):
            timestamps = np.hstack((timestamps, [np.datetime64(end_window)])) 
            values_act_win = np.hstack((values_act_win, [ts_act.values[cnt_end-1]]))
                    
        # Now, shift the actual timeseries so as the first and last timestamps 
        # match with the estimated timeseries        
        ts_act_win = pd.Series(values_act_win, pd.DatetimeIndex(timestamps - np.timedelta64(start_window - utils.START_DATETIME)))
                 
        # Take the union of all available timestamps         
        timestamps_all = pd.DatetimeIndex(np.union1d(ts_act_win.index, ts_est.index))
        
        # Fit both timeseries to the timestamps union
        ts_act_win = ts_act_win.reindex(timestamps_all, method='pad')
        ts_est_win = ts_est.reindex(timestamps_all, method='pad')

        # Now, compute the distance         
        dist = utils.distance_midi_pitch(ts_act_win.values, ts_est_win.values)
        
        # Append the results
        timestamps_dist.append(start_window)
        values_dist.append(dist) 
        
        # Move forward by hop_s seconds
        start_window += dt.timedelta(0, hop_s, 0) 
        end_window += dt.timedelta(0, hop_s, 0)
        
    
    # Reshape the data as a timeseries and interpolate flat (for plotting)
    ts_dist = pd.Series(values_dist, timestamps_dist) 
    ts_dist = ts_dist.add(ts_dist.shift(1).shift(-1, freq = utils.MIN_DELTATIME_STR), fill_value=0)
        
    return(ts_dist)

def compare_act_and_est_snippet2(ts_act, ts_est):
    """
    Given the time series of the actual midi file and an estimate of the pitches
    estimated with Aubio, find where we are likely to be in the midi file 
    
    Second version of the function, it resamples both timeseries at a higher 
    frequency and computes the distance, naturally using the equal spacing. 
    """
    
    # Start by replacing the NaNs with zeros
    # Round the values to the nearest integer
    ts_act = ts_act.replace(np.NaN, 0.0).apply(round)
    ts_est = ts_est.replace(np.NaN, 0.0).apply(round)
    
    # Resample both timeseries
    resample_freq = '10ms'
    ts_act_resampled = ts_act.resample(resample_freq, fill_method='pad')
    ts_est_resampled = ts_est.resample(resample_freq, fill_method='pad')
    
    nb_obs_est = len(ts_est_resampled.index)
    nb_obs_act = len(ts_act_resampled.index)    
    
    # Compute the distance, rolling forward on ts_act    
    values_dist = []
    for k in range(nb_obs_est,nb_obs_act): # maybe jump more that 1 k at a time?
        dist = utils.distance_midi_pitch(ts_act_resampled.values[k-nb_obs_est:k], ts_est_resampled.values)
        values_dist.append(dist)
        
    # Build the distance timeseries 
    timestamps_dist = ts_act_resampled.index[0:nb_obs_act-nb_obs_est]  
    ts_dist = pd.Series(values_dist, timestamps_dist) 
    ts_dist = ts_dist.add(ts_dist.shift(1).shift(-1, freq = utils.MIN_DELTATIME_STR), fill_value=0)
    
    return(ts_dist)


# Input the files names used for the test
wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Badinerie/"
filename = "badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"
filename_midi = wd + filename + ".mid"

# Process the "true" midi values 
ts_act = utils.process_midi_file(filename_midi)

# Estimate the midi values from the .wav file for a specific time lapse
ts_est = utils.process_wav_file(filename_wav, start_sec = 48.0, end_sec = 51.0)

# Find the distance between the true midi and the estimated midi
start = timer() 
ts_dist = compare_act_and_est_snippet(ts_act, ts_est)
# ts_dist = compare_act_and_est_snippet2(ts_act, ts_est)
end = timer()
print(end - start)

# # Plot
# midi_range = (65.0, 90.0)
# plt.subplot(311)
# ts_act.plot()
# plt.legend(['Midi File Pitch'])
# axis1 = plt.axis()
# plt.axis([axis1[0], axis1[1], midi_range[0], midi_range[1]])
# plt.subplot(312)
# ts_est.plot()
# plt.axis([axis1[0], axis1[1], midi_range[0], midi_range[1]])
# plt.legend(['Estimated Pitch'])
# plt.subplot(313)
# ts_dist.plot()
# plt.legend(['Distance'])
# plt.show()
