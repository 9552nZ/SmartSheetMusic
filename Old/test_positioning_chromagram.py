'''
The script: 
1) Writes (or loads) the "true" spectrogram
2) Turns it into a chromagram
3) Compares it with another chromagram sample
4) Plots the distance between the "true" and the estimated chromagram
'''

import numpy as np
import matplotlib.pyplot as plt
import utils_audio_transcript as utils

SAMPLERATE = 11025
WIN_S = 4096
HOP_S = 2048

wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Nocturnes/"
filename = "nocturnes"#"badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"
filename_spectrogram = utils.get_filename_spectrogram(wd + filename, SAMPLERATE, WIN_S, HOP_S)


# utils.write_spectrogram_to_disk(wd, filename, SAMPLERATE, WIN_S, HOP_S)

spec_data_act = np.load(filename_spectrogram).item()
chromagram_act = utils.spectrogram_to_chromagram(spec_data_act["spectrogram"], spec_data_act["frequencies"])
  
spec_data_est = utils.get_wav_spectrogram(filename_wav, samplerate=SAMPLERATE, win_s=WIN_S, hop_s=HOP_S, start_sec=21.0, end_sec=26.0)
chromagram_est = utils.spectrogram_to_chromagram(spec_data_est["spectrogram"], spec_data_est["frequencies"])
  
ts_dist = utils.compare_chomagrams(chromagram_act, chromagram_est, spec_data_est["samplerate"], spec_data_est["hop_s"])
  
ts_dist.plot()
plt.show()
