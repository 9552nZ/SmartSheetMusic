import matplotlib.pyplot as plt
import utils_audio_transcript as utils

"""
The script:
1) Opens a music file stored as a .wav
3) Uses Aubio to estimate the pitches
4) Compares the estimated pitch with the real pitches as stored in a midi file
5) Plots the two time series
"""
# Input the files names used for the test
wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Badinerie/"
filename = "badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"
filename_midi = wd + filename + ".mid"

ts_act = utils.process_midi_file(filename_midi)
ts_est = utils.process_wav_file(filename_wav)

# Lead the curve for a bit as it seems there is a bit of lag introduced somewhere...
# ts_est = ts_est.shift(-1, freq='80ms') 

# Now plot
ts_act.plot()
ts_est.plot()
plt.legend(['Midi File Pitch', 'Estimated Pitch'])
plt.show()