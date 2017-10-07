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
import zmq
from time import sleep
import utils_audio_transcript as utils
from score_following import Matcher
from music_detection import MusicDetecter
import os # REMOVE
import datetime # REMOVE

class MatcherManager():
    def __init__(self, filename_mid_full):
        self.chunk = 16384/16
        self.sr = utils.SR
        self.hop_length = utils.HOP_LENGTH
        self.audio_format = utils.AUDIO_FORMAT_DEFAULT # The audio low-level format
        self.detect_music = False # Set to true to only start once the MLP has detected music
         
        (wd, filename) = os.path.split(filename_mid_full)
        wd = wd + "/"
        
        # Initialise the placeholder for the estimated audio data (only the most recent)
        # This is only used in the online mode, as we may want to compute the chromagram  
        # with more data that just the last online chunk.
        self.audio_data = [] 
        self.min_len_chromagram_sec = 4.0 # Seconds of audio we want to compute the chromagram
        self.min_len_sample = utils.calc_nb_sample_stft(self.sr, self.hop_length, self.min_len_chromagram_sec) # Nb of samples to compute the chromagram 
     
        # Initialialise the matcher
        self.matcher = Matcher(wd, filename, self.sr, self.hop_length)
     
        # Initialise the music detecter
        if self.detect_music:
            self.music_detecter = MusicDetecter(utils.WD_AUDIOSET, self.sr, self.hop_length, 84, self.min_len_sample)
        
        # Start the publishing socket
        self.bind_sockets()
        
        # Start the pyaudio stream 
        self.start_stream()
        
        self.recording = False
 
    def bind_sockets(self):
        # Start the ZMQ sockets
        context = zmq.Context()
        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.bind("tcp://*:5555")
        self.socket_rep = context.socket(zmq.REP)
        self.socket_rep.bind("tcp://*:5556")
        
    def start_stream(self):
        '''
        Start the pyaudio stream
        '''
        self.pyaudio = pyaudio.PyAudio()
    
        self.stream = self.pyaudio.open(format = utils.AUDIO_FORMAT_MAP[self.audio_format][1],                                   
                                        channels = 1, # TODO: Make sure that everything is built to process mono.
                                        rate = self.sr,
                                        input = True,
                                        frames_per_buffer = self.chunk, 
                                        stream_callback = callback)
        
        self.stream.start_stream() # Not needed, but keep it to be sure.
        
    def stop_stream(self): 
        '''
        Stop the pyaudio stream
        '''   
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        
    def update_audio_data(self, new_audio_data):
        '''        
        Convert the new audio data to the relevant format, append it to the 
        existing data and only keep what we need. 
        '''
        
        # Start by converting the input string to numpy array
        new_audio_data = np.fromstring(new_audio_data, dtype=utils.AUDIO_FORMAT_MAP[self.audio_format][0])
        
        # Convert to a format that can be processed by librosa
        new_audio_data = new_audio_data.astype(np.float32).tolist()
        
        # Make sure that the chunk size is a multiple of hop size
        if (len(new_audio_data) % self.hop_length) != 0:
            raise ValueError('Chunk size need to be a multiple of hop size')
        
        self.audio_data.extend(new_audio_data)      
        self.audio_data = self.audio_data[max(len(self.audio_data) - self.min_len_sample, 0):len(self.audio_data)]
        
    def check_ready_for_start(self):
        '''
        Check if we should start the matching procedure.
        The function ensures that music has been detected.        
        '''
        ready = True
        if len(self.audio_data) < self.min_len_sample:
            ready = False
            
        if self.detect_music:
            ready = self.music_detecter.detect(self.audio_data)
        
        return(ready)
        
    def callback(self, new_audio_data):
        '''
        The callback, called for every CHUNK of new data.
        Non-blocking callback.
        Process CHUNK samples of data, updating the matcher.
        
        '''
        if not self.recording:
            # Block until we receive the starting instruction from the GUI
            self.socket_rep.recv()
            self.socket_rep.send(b"")
            self.recording = True        
        
        # Once we started recording, keep the latest audio data
        self.update_audio_data(new_audio_data)
        
        if not self.check_ready_for_start():
            return        
                        
        # We compute the chromagram only for CHUNK samples.
        # Not sure this is ideal, we may want to keep some history of the audio 
        # input and compute the chromagram based on that. 
        chromagram_est = Matcher.compute_chromagram(self.audio_data,
                                           self.matcher.sr_act,
                                           self.matcher.hop_length_act,                                             
                                           self.matcher.compute_chromagram_fcn, 
                                           self.matcher.compute_chromagram_fcn_kwargs,
                                           self.matcher.chromagram_mode)
                                                             
        print "Time: {}   Size chromagram:{}    Size audio: {}".format(datetime.datetime.now().time(), chromagram_est.shape,  len(self.audio_data))
        
        # Only run the matching over the last (len(new_audio_data)/self.hop_length) segments of the chromagram
        idx_frames = np.arange(chromagram_est.shape[0] - len(new_audio_data) / self.hop_length, chromagram_est.shape[0])
        
        # Run the online matching procedure, one frame at a time
        for n in idx_frames:
            self.matcher.main_matcher(chromagram_est[n,:])   
        
    def publish_position(self):
        '''
        Send the current position to the GUI via the socket.
        '''
        print "Sending current position: {}".format(self.matcher.position_tick)
        self.socket_pub.send(bytes(self.matcher.position_tick))                               
 
    def plot_chromagram(self):
        '''
        Use the function for online plotting of the chromagram.
        '''
        import matplotlib.pyplot as plt
        
        if len(self.music_detecter.features):
            if plt.get_fignums():
                self.pcol = utils.plot_chromagram(self.music_detecter.features.T, pcol=self.pcol)
            else:
                self.pcol = utils.plot_chromagram(self.music_detecter.features.T, sr=self.music_detecter.mlp.sr, hop_length=self.music_detecter.mlp.hop_length)            
            
def callback(in_data, frame_count, time_info, flag):
    global matcher_manager
    matcher_manager.callback(in_data)
    return (None, pyaudio.paContinue)
            
if __name__ == '__main__':
    
    # Input the target '.mid' file of the score that we wish to follow 
    if len(sys.argv) > 1:
        filename_mid = sys.argv[1]
    else:
        filename_mid = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.mid"

    matcher_manager = MatcherManager(filename_mid)    
    
    while matcher_manager.stream.is_active():
#         matcher_manager.plot_chromagram()
        matcher_manager.publish_position()
        sleep(matcher_manager.chunk/float(matcher_manager.sr))  
    