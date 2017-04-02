import pyaudio
import numpy as np
import os
import utils_audio_transcript as utils
from aubio import pitch
import matplotlib.pyplot as plt

"""
This script:
1) Plays a music file
2) Has the micro listening to it
3) Uses Aubio to guess what pitches are being played
4) Compares the estimated pitch with the real pitches as stored in a midi file
5) Plots the two time series
"""

wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Badinerie/"
filename = "badinerie_bach_flute" 
filename_wav = wd + filename + ".wav"
filename_midi = wd + filename + ".mid"


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 20

os.system("start " + filename_wav)
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

# Pitch
tolerance = 0.8
downsample = 1
win_s = 4096 // downsample # fft size
# hop_s = 1024  // downsample # hop size
hop_s = CHUNK 
pitch_o = pitch("yin", win_s, hop_s, RATE)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches_est = []
times_est = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    buffer = stream.read(CHUNK)
    frames.append(buffer)

    signal = np.fromstring(buffer, dtype=np.float32)

    pitch = pitch_o(signal)[0]
    pitches_est += [pitch]
    times_est += [i * CHUNK / float(RATE)]

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

ts_est = utils.relative_ts_to_absolute_ts(times_est, pitches_est)
ts_act = utils.process_midi_file(filename_midi)

ts_act.plot()
ts_est.plot()
plt.legend(['Midi File Pitch', 'Estimated Pitch'])
plt.show()

