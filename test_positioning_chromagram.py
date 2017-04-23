import numpy as np
import matplotlib.pyplot as plt
import utils_audio_transcript as utils

def compare_chomagrams(chromagram_act, chromagram_est, samplerate, hop_s):
    '''
    Compute the distance between an estimated chromagram (proecssed online) and a 
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
        dist += [utils.distance_chroma(chromagram_act[(k-nb_frames_est):k,], chromagram_est, nb_elmts)]
        
    # Reshape the distance as a timeseries
    times = np.arange(0, nb_frames_act-nb_frames_est) * hop_s / float(samplerate)
        
    # Convert to absolute timestamps
    ts_dist = utils.relative_ts_to_absolute_ts(times, dist)
    
    return(ts_dist)

wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Badinerie/"
filename = "badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"

filename_chroma = wd + filename + "_chroma.npy"
# spec_data_act = get_spectrogram(filename_wav)
# np.save(filename_chroma, spec_data_act)
spec_data_act = np.load(filename_chroma).item()
chromagram_act = utils.spectrogram_to_chromagram(spec_data_act["specgram"], spec_data_act["frequencies"])

spec_data_est = utils.get_spectrogram(filename_wav, start_sec = 48.0, end_sec = 51.0)
chromagram_est = utils.spectrogram_to_chromagram(spec_data_est["specgram"], spec_data_est["frequencies"])

ts_dist = compare_chomagrams(chromagram_act, chromagram_est, spec_data_est["samplerate"], spec_data_est["hop_s"])

ts_dist.plot()
plt.show()