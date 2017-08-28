import pyaudio
import numpy as np
import sys
import os
from time import sleep
from score_following import Matcher


def callback(in_data, frame_count, time_info, flag):
    
    global matcher # The callback function cannot take extra arguments
    
    audio_data = np.fromstring(in_data, dtype=np.float32).tolist()
    
    # We compute the chromagram only for CHUNK samples.
    # Not sure this is ideal, we may want to keep some history of the audio input and compute the
    # chromagram based on that. 
    chromagram_est = Matcher.compute_chromagram(audio_data,
                                                matcher.sr_est,
                                                matcher.hop_length_est,                                             
                                                matcher.compute_chromagram_fcn, 
                                                matcher.compute_chromagram_fcn_kwargs,
                                                matcher.chromagram_mode)                                                         
    
    # Run the online alignment, frame by frame
    nb_frames_est = chromagram_est.shape[0]
    for n in range(nb_frames_est):
        matcher.main_matcher(chromagram_est[n,:])
        
    return (matcher.position_sec[-1], pyaudio.paContinue)            

if __name__ == '__main__':
    
    CHUNK = 16384
    SR = 11025
    HOP_LENGTH = 1024
    
    full_filename = sys.argv[1]
    (wd, filename) = os.path.split(full_filename)
    wd = wd + "/"
    
    matcher = Matcher(wd, filename, SR, HOP_LENGTH) 
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format = pyaudio.paFloat32,
                channels = 1, # TODO: Make sure that everything is built to process mono.
                rate = SR,
                input = True,
                frames_per_buffer = CHUNK, 
                stream_callback = callback)
     
    stream.start_stream()
     
    while stream.is_active():
        sleep(0.1)        
     
    stream.stop_stream()
    stream.close()
    p.terminate()

