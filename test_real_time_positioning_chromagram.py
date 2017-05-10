'''
The script listens in real time what is being played and tries to find the 
corresponding position in the  MIDI file
'''

import pyaudio
import numpy as np
import time
from real_time_positioning import Position


def callback(in_data, frame_count, time_info, flag):
    '''
    Each time the system gets a new chunk of data, this function is called.
    We try to find the best position in the MIDI using the new piece of information.
    '''
    global pos
    
    audio_data = np.fromstring(in_data, dtype=np.float32).tolist()
    pos.find_position(audio_data)    
    
    return(('', False))

wd = "C:/Users/Alexis/Desktop/Business/SmartSheetMusic/Samples/Nocturnes/"
filename = "nocturnes" 
filename_wav = wd + filename + ".wav"
filename_midi = wd + filename + ".mid"

CHUNK = 16384
SAMPLERATE = 11025
WIN_S = 4096
HOP_S = 2048

pos = Position(wd, filename, SAMPLERATE, WIN_S, HOP_S, verbose = True, plot_online = False)

p = pyaudio.PyAudio()
 
stream = p.open(format = pyaudio.paFloat32,
                channels = 1,
                rate = SAMPLERATE,
                input = True,
                frames_per_buffer = CHUNK, 
                stream_callback = callback)
 
print("* recording")
 
stream.start_stream()
 
while stream.is_active():
    time.sleep(0.1)
    
print("* done recording")
 
stream.stop_stream()
stream.close()
p.terminate()

