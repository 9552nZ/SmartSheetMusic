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

# Flag for the communication with the GUI
STOP = "stop"
START = "start"

class MatcherManager():
    ''' 
    Higher-level class used to manage the interactions between: 
    - the matcher
    - the pyaudio stream
    - the zmq sockets
    - the music detector
    '''
    def __init__(self, filename_mid_full):
        self.chunk = 16384/16 # The chunk of data processed per pyaudio batch
        self.sr = utils.SR # Sample rate
        self.hop_length = utils.HOP_LENGTH 
        self.audio_format = utils.AUDIO_FORMAT_DEFAULT # The audio low-level format
        self.detect_music = False # Set to true to only start once the MLP has detected music
         
        # Split the path of the midi input between the directory and file name.
        (wd, self.filename) = os.path.split(filename_mid_full)
        self.wd = wd + "/"
        
        # Initialise the placeholder for the estimated audio data (only the most recent)
        # This is only used in the online mode, as we may want to compute the chromagram  
        # with more data that just the last online chunk.
        self.audio_data = [] 
        self.min_len_chromagram_sec = 4.0 # Seconds of audio we want to compute the chromagram
        self.min_len_sample = utils.calc_nb_sample_stft(self.sr, self.hop_length, self.min_len_chromagram_sec) # Nb of samples to compute the chromagram 
     
        # Initialialise the matcher
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length)
     
        # Initialise the music detecter
        if self.detect_music:
            self.music_detecter = MusicDetecter(utils.WD_AUDIOSET, self.sr, self.hop_length, 84, self.min_len_sample)
        
        # Start the publishing socket
        self.bind_sockets()
        
        # Start the pyaudio stream 
        self.start_stream()
        
        # We initialise with a STOP status  
        self.status = STOP
        
    def reinit(self):
        '''
        Reinitialise the audio data.
        Reinitialise the matcher, using the already computed chromagram.
        ''' 
        self.audio_data = []
        midi_obj = self.matcher.midi_obj
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length, chromagram_act=self.matcher.chromagram_act)
        self.matcher.midi_obj = midi_obj 
        self.status = STOP
 
    def bind_sockets(self):
        '''
        Start the ZMQ sockets
        '''
        context = zmq.Context()
        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.bind("tcp://*:5555")
        self.socket_sub = context.socket(zmq.SUB)        
        self.socket_sub.setsockopt(zmq.SUBSCRIBE, '')
        self.socket_sub.connect("tcp://localhost:5556")
        
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
    
    def callback_main(self, new_audio_data):
        '''
        The callback, called for every CHUNK of new data.
        Non-blocking callback.
        Process CHUNK samples of data, updating the matcher.
        
        '''         
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
                
    def callback(self, new_audio_data):
        '''
        The higher-level pyaudio callback.
        We start by checking if anything START/STOP signal has come through the 
        socket.
        We can either:
        - do nothing, if we are stopped and the GUI says to remain so.
        - send to callback_main if the GUI says to start
        - reinitialise if we are started and the GUI says to stop.
        '''
        try:
            flag = self.socket_sub.recv(flags=zmq.NOBLOCK) # What happens when we have multiple messages?? Maybe set the SUB to CONFLATE                                           
        except zmq.Again:
            flag = self.status
        
        if self.status == STOP and flag == STOP:
            pass
        elif (self.status == START and flag == START) or (self.status == STOP and flag == START): 
            self.callback_main(new_audio_data)
            self.status = START
        elif self.status == START and flag == STOP:
            self.reinit()
            self.status = STOP
        else:
            raise(ValueError('Unexpected status ({}) or flag ({})'.format(self.status, flag)))                    
                    
            
def callback(in_data, frame_count, time_info, flag):
    '''
    We need to have a callback outside the class.
    This function is called in a  new thread.
    It could be a problem if there are high fixed cost to copy the matcher in the new thread.    
    '''
    global matcher_manager
    matcher_manager.callback(in_data)
    return (None, pyaudio.paContinue)
            
if __name__ == '__main__':
     
    # Input the target '.mid' file of the score that we wish to follow 
    if len(sys.argv) > 1:
        filename_mid = sys.argv[1]
    else:
        filename_mid = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/Chopin_Op028-01_003_20100611-SMD.mid"
     
    try:
        matcher_manager = MatcherManager(filename_mid)    
         
        while matcher_manager.stream.is_active():
    #         matcher_manager.plot_chromagram()
            matcher_manager.publish_position()
            sleep(matcher_manager.chunk/float(matcher_manager.sr))  
    except:
        sleep(100.0)