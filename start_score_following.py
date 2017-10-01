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
        self.bind_socket()
        
        # Start the pyaudio stream 
        self.start_stream()
 
    def bind_socket(self):
        # Start the publishing ZMQ socket
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")
        
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
        !!!! NOT PLUGGED IN FOR NOW !!!
        '''
        
        if self.detect_music:
            music_detected = self.music_detecter.detect(self.audio_data)
        
        return(True)
        
    def callback(self, new_audio_data):
        '''
        The callback, called for every CHUNK of new data.
        Non-blocking callback.
        Process CHUNK samples of data, updating the matcher.
        
        '''
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
                                                             
#         print "Time: {}   Size chromagram:{}    Size audio: {}".format(datetime.datetime.now().time(), chromagram_est.shape,  len(self.audio_data))
        
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
        self.socket.send(bytes(self.matcher.position_tick))                               
 
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

# def update_audio_data(new_audio_data, audio_data, keep_length):
#     '''
#     Receive new audio data, append to existing and only keep the required 
#     amount of most recent data.
#     '''
#     audio_data.extend(new_audio_data)      
#     audio_data = audio_data[max(len(audio_data) - keep_length, 0):len(audio_data)]
#     
#     return audio_data
# 
# def callback(in_data, frame_count, time_info, flag):
#     '''
#     Non-blocking callback. 
#     Process CHUNK samples of data, updating the matcher.
#     '''
#     # The callback function cannot take extra arguments so we 
#     # need global variables.
#     global matcher 
#     global music_detecter
#     
#     # Start by converting the input data
#     in_data = np.fromstring(in_data, dtype=np.int16).astype(np.float32).tolist()
#     
#     # Make sure that the chunk size is a multiple of hop size
#     if (len(in_data) % matcher.hop_length_act) != 0:
#         raise ValueError('Chunk size need to be a multiple of hop size')
#     
#     # Keep only the required amout of data
#     audio_data_est = update_audio_data(in_data, matcher.audio_data_est, matcher.min_len_sample)
#     
#     matcher.audio_data_est = audio_data_est
#     
#     music_detected = music_detecter.detect(audio_data_est)
#         
#     # We compute the chromagram only for CHUNK samples.
#     # Not sure this is ideal, we may want to keep some history of the audio input and compute the
#     # chromagram based on that. 
#     chromagram_est = Matcher.compute_chromagram(matcher.audio_data_est,
#                                                 matcher.sr_act,
#                                                 matcher.hop_length_act,                                             
#                                                 matcher.compute_chromagram_fcn, 
#                                                 matcher.compute_chromagram_fcn_kwargs,
#                                                 matcher.chromagram_mode)
#                                                                   
#     print "Time: {}   Size chromagram:{}    Size audio: {}".format(datetime.datetime.now().time(), chromagram_est.shape,  len(matcher.audio_data_est))
#        
#     frames = np.arange(chromagram_est.shape[0] - len(in_data) / matcher.hop_length_act, chromagram_est.shape[0])
#      
#     # Run the online matching procedure, one frame at a time
#     for n in frames:
#         matcher.main_matcher(chromagram_est[n,:])
#              
#     return (None, pyaudio.paContinue)
# 
# if __name__ == '__main__':
#     plt.ion() # REMOVE
#     CHUNK = 16384/16
#     sr = utils.SR
#     hop_length = utils.HOP_LENGTH
# #     HOP_LENGTH = 1024
#     min_len_chromagram_sec = 4
#     
#     # Input the target '.mid' file of the score that we wish to follow 
#     if len(sys.argv) > 1:
#         full_filename = sys.argv[1]
#     else:
#         full_filename = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.mid"
#     
#         
#     (wd, filename) = os.path.split(full_filename)
#     wd = wd + "/"
#     
#     # Initialialise the matcher
#     matcher = Matcher(wd, filename, sr, hop_length, min_len_chromagram_sec=min_len_chromagram_sec)
#     
#     # Initialise the music detecter
#     music_detecter = MusicDetecter(utils.WD_AUDIOSET, sr, hop_length, 84, matcher.min_len_sample)
#     
#     # Start the publishing socket
#     context = zmq.Context()
#     socket = context.socket(zmq.PUB)
#     socket.bind("tcp://*:5555")
#     
#     # Start the .wav file. !!!! REMOVE !!!!  
# #     proc = subprocess.Popen(["C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.wav"], 
# #                             shell=True,
# #                             stdin=None, stdout=None, stderr=None, close_fds=True)   
#     
#     # Start the pyaudio stream 
#     p = pyaudio.PyAudio()
#     
#     stream = p.open(format = pyaudio.paInt16,
#                 channels = 1, # TODO: Make sure that everything is built to process mono.
#                 rate = sr,
#                 input = True,
#                 frames_per_buffer = CHUNK, 
#                 stream_callback = matcher_manager.callback)
#     
# #     stream.start_stream()
#     first_plot = True
#     # Publish the current position every CHUNK/float(SR) seconds
#     
#     while stream.is_active():
# #         print "Sending current position: {}".format(matcher.position_tick)
# #         socket.send(bytes(matcher.position_tick))    

# 
#         sleep(0.1)
# #         sleep(CHUNK/float(sr))        
     
#     stream.stop_stream()
#     stream.close()
#     p.terminate()


    