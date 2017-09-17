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
from score_following import Matcher
from math import ceil
import subprocess
import datetime


def update_audio_data(new_audio_data, audio_data, keep_length):
    '''
    Receive new audio data, append to existing and only keep the required 
    amount of most recent data.
    '''
    audio_data.extend(new_audio_data)  
    audio_data = audio_data[len(audio_data) - keep_length:len(audio_data)]
    
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
    in_data = np.fromstring(in_data, dtype=np.float32).tolist()
    
    # Make sure that the chunck size is a multiple of hop size
    if (len(in_data) % matcher.hop_length_act) != 0:
        raise ValueError('Chunck size need to be a multiple of hop size')
    
#     # Convert the input data to numpy array
#     audio_data = np.array(np.fromstring(in_data, dtype=np.float32).tolist())
    keep_length = int(ceil(matcher.sr_act*matcher.min_len_chromagram_sec/matcher.hop_length_act)) * matcher.hop_length_act
    matcher.audio_data_est = update_audio_data(in_data, matcher.audio_data_est, keep_length) 
    
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
      
    nb_frames_est = len(in_data) / matcher.hop_length_act
    frames = np.arange(chromagram_est.shape[0] - nb_frames_est, chromagram_est.shape[0])
    
    #!!!!!!!!!!!! COMMENT !!!!!!!!!!!!!!
    for n in frames:
        matcher.main_matcher(chromagram_est[n,:])
             
    return (None, pyaudio.paContinue)

if __name__ == '__main__':
    CHUNK = 16384/16
    SR = 11025
    HOP_LENGTH = 1024
    min_len_chromagram_sec = 4
    
    # Input the target '.mid' file of the score that we wish to follow 
    if len(sys.argv) > 1:
        full_filename = sys.argv[1]
    else:
        full_filename = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.mid"
    
        
    (wd, filename) = os.path.split(full_filename)
    wd = wd + "/"
    
    # Initialialise the matcher
    matcher = Matcher(wd, filename, SR, HOP_LENGTH, min_len_chromagram_sec=min_len_chromagram_sec)
    
    # Start the publishing socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    
    # Start the .wav file. !!!! REMOVE !!!!  
    proc = subprocess.Popen(["C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.wav"], 
                            shell=True,
                            stdin=None, stdout=None, stderr=None, close_fds=True)   
    
    # Start the pyaudio stream 
    p = pyaudio.PyAudio()
    
    stream = p.open(format = pyaudio.paFloat32,
                channels = 1, # TODO: Make sure that everything is built to process mono.
                rate = SR,
                input = True,
                frames_per_buffer = CHUNK, 
                stream_callback = callback)
    
    stream.start_stream()
    
    # Publish the current position every CHUNK/float(SR) seconds
    while stream.is_active():
        print "Sending current position: {}".format(matcher.position_tick)
        socket.send(bytes(matcher.position_tick))    
        sleep(CHUNK/float(SR))        
     
    stream.stop_stream()
    stream.close()
    p.terminate()


    