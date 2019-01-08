import pyaudio
import numpy as np
import sys
from librosa.output import write_wav as lb_write_wav
import librosa as lb
from time import sleep
import utils_audio_transcript as utils
import music_detection
from music_detection import MusicDetecter
from pandas import DataFrame
import music_detection
from noise_reduction import NoiseReducer
import matplotlib.pyplot as plt
from keras.utils import plot_model


def callback(in_data, frame_count, time_info, flag):
    
    global audio_data
    
    audio_data_new = np.fromstring(in_data, dtype=utils.AUDIO_FORMAT_MAP[audio_format][0])
    audio_data = np.append(audio_data, audio_data_new)   
    audio_data_tmp = audio_data[max(len(audio_data)-music_detecter.nb_sample, 0):len(audio_data)]
    
    music_detected, music_detection_diagnostic = music_detecter.detect(audio_data_tmp)
        
    print("Music detected: {}   Diagnostic: {}".format(music_detected, music_detection_diagnostic))    
    
    return (None, pyaudio.paContinue)

chunk = int(4096)         
sr = utils.SR 
hop_length = utils.HOP_LENGTH 
audio_format = utils.AUDIO_FORMAT_DEFAULT  
audio_data = np.array([])
music_detecter = MusicDetecter()
# noise_reducer = NoiseReducer()
 
pyaudio_new = pyaudio.PyAudio()
 
stream = pyaudio_new.open(format = utils.AUDIO_FORMAT_MAP[audio_format][1],
                                  channels = 1,
                                  rate = sr,
                                  input = True,
                                  frames_per_buffer = chunk, 
                                  stream_callback = callback)
 
stream.start_stream()
 
while True:
    sleep(0.1)
                
stream.stop_stream()
stream.close()
