'''
This scripts is the entry point of the online score following.
It is called by the graphical client, and takes as input the path of the '.mid' file.
1) initialise the matcher  
2) initialise a socket to publish the results of the matching procedure
3) start a pyaudio stream
4) for each new chunk of data:
    - run the online matching
    - publish the results via the socket
'''

import pyaudio
import numpy as np
import sys
import os
import zmq
from time import sleep
import subprocess
import datetime
import utils_audio_transcript as utils
from score_following import Matcher
from music_detection import MusicDetecter
import os # REMOVE
import matplotlib.pyplot as plt # REMOVE


class MatcherStarter():
    def __init__(self):
        

def plot_chromagram():
    if plt.get_fignums():


def update_audio_data(new_audio_data, audio_data, keep_length):
    '''
    Receive new audio data, append to existing and only keep the required 
    amount of most recent data.
    '''
    audio_data.extend(new_audio_data)      
    audio_data = audio_data[max(len(audio_data) - keep_length, 0):len(audio_data)]
    
    return audio_data

def callback(in_data, frame_count, time_info, flag):
    '''
    Non-blocking callback. 
    Process CHUNK samples of data, updating the matcher.
    '''
    # The callback function cannot take extra arguments so we 
    # need global variables.
    global matcher 
    global music_detecter
    
    # Start by converting the input data
    in_data = np.fromstring(in_data, dtype=np.int16).astype(np.float32).tolist()
    
    # Make sure that the chunk size is a multiple of hop size
    if (len(in_data) % matcher.hop_length_act) != 0:
        raise ValueError('Chunk size need to be a multiple of hop size')
    
    # Keep only the required amout of data
    audio_data_est = update_audio_data(in_data, matcher.audio_data_est, matcher.min_len_sample)
    
    matcher.audio_data_est = audio_data_est
    
    music_detected = music_detecter.detect(audio_data_est)
        
    # We compute the chromagram only for CHUNK samples.
    # Not sure this is ideal, we may want to keep some history of the audio input and compute the
    # chromagram based on that. 
    chromagram_est = Matcher.compute_chromagram(matcher.audio_data_est,
                                                matcher.sr_act,
                                                matcher.hop_length_act,                                             
                                                matcher.compute_chromagram_fcn, 
                                                matcher.compute_chromagram_fcn_kwargs,
                                                matcher.chromagram_mode)
                                                                  
    print "Time: {}   Size chromagram:{}    Size audio: {}".format(datetime.datetime.now().time(), chromagram_est.shape,  len(matcher.audio_data_est))
       
    frames = np.arange(chromagram_est.shape[0] - len(in_data) / matcher.hop_length_act, chromagram_est.shape[0])
     
    # Run the online matching procedure, one frame at a time
    for n in frames:
        matcher.main_matcher(chromagram_est[n,:])
             
    return (None, pyaudio.paContinue)

if __name__ == '__main__':
    plt.ion() # REMOVE
    CHUNK = 16384/16
    sr = utils.SR
    hop_length = utils.HOP_LENGTH
#     HOP_LENGTH = 1024
    min_len_chromagram_sec = 4
    
    # Input the target '.mid' file of the score that we wish to follow 
    if len(sys.argv) > 1:
        full_filename = sys.argv[1]
    else:
        full_filename = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.mid"
    
        
    (wd, filename) = os.path.split(full_filename)
    wd = wd + "/"
    
    # Initialialise the matcher
    matcher = Matcher(wd, filename, sr, hop_length, min_len_chromagram_sec=min_len_chromagram_sec)
    
    # Initialise the music detecter
    music_detecter = MusicDetecter(utils.WD_AUDIOSET, sr, hop_length, 84, matcher.min_len_sample)
    
    # Start the publishing socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    
    # Start the .wav file. !!!! REMOVE !!!!  
#     proc = subprocess.Popen(["C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.wav"], 
#                             shell=True,
#                             stdin=None, stdout=None, stderr=None, close_fds=True)   
    
    # Start the pyaudio stream 
    p = pyaudio.PyAudio()
    
    stream = p.open(format = pyaudio.paInt16,
                channels = 1, # TODO: Make sure that everything is built to process mono.
                rate = sr,
                input = True,
                frames_per_buffer = CHUNK, 
                stream_callback = callback)
    
#     stream.start_stream()
    first_plot = True
    # Publish the current position every CHUNK/float(SR) seconds
    
    while stream.is_active():
#         print "Sending current position: {}".format(matcher.position_tick)
#         socket.send(bytes(matcher.position_tick))    
        if len(music_detecter.features):
            if 'pcol' not in locals():
                pcol = utils.plot_chromagram(music_detecter.features.T, sr=music_detecter.mlp.sr, hop_length=music_detecter.mlp.hop_length)
            else:
                pcol = utils.plot_chromagram(music_detecter.features.T, pcol=pcol)

        sleep(0.1)
#         sleep(CHUNK/float(sr))        
     
    stream.stop_stream()
    stream.close()
    p.terminate()


    