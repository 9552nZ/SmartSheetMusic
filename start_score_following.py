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
# from pandas import DataFrame
import os # REMOVE
# import datetime # REMOVE
# import multiprocessing # REMOVE

# Flag for the communication with the GUI
# TODO : Replace by integers
STOP = b"stop"
START = b"start"
MATCH = b"match"
PAUSE = b"pause"
INIT = b"init" 

class MatcherManager():
    ''' 
    Higher-level class used to manage the interactions between: 
    - the matcher
    - the pyaudio stream
    - the zmq sockets
    - the music detector
    '''
    def __init__(self, filename_mid_full, callback_fcn):
        self.chunk = int(16384/16) # The chunk of data processed per pyaudio batch        
        self.sr = utils.SR # Sample rate
        self.hop_length = utils.HOP_LENGTH 
        self.audio_format = utils.AUDIO_FORMAT_DEFAULT # The low-level audio format 
        self.detect_music = True # Set to true to only start once the MLP has detected music
        self.callback_fcn = callback_fcn # The pyaudio callback function  
        
        # Start the publishing socket
        self.bind_sockets()
         
        # Split the path of the midi input between the directory and file name.
        (wd, self.filename) = os.path.split(filename_mid_full)
        self.wd = wd + "/" 
     
        # Initialise the matcher
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length, 
                               compute_chromagram_fcn_kwargs={'n_fft':self.hop_length*2, 'n_chroma':12})                    
     
        # Initialise the music detecter
        self.music_detection_diagnostic = ''
        self.music_detected_all = []
        if self.detect_music:
            self.music_detecter = MusicDetecter()              
            
        # Initialise the audio data buffer
        self.audio_data = np.array([])
        
        # Start the pyaudio stream 
        self.start_stream()
        
        # We initialise with a STOP status for the GUI and a PAUSE for the matcher         
        self.status = STOP
        self.status_matcher = PAUSE                         
        
    def __getstate__(self):
        '''
        Called by pickle.dump to know what attributes need to be pickled. 
        Pickle everything but the sockets and pyaudio.
        '''
        keys_not_picklable = ['socket_pub', 'socket_sub', 'socket_req_init', 'pyaudio', 'stream', 'callback_fcn']
        return {k: v for k, v in self.__dict__.items() if k not in keys_not_picklable}
    
        
    def reinit(self):
        '''
        Reinitialise the audio data.
        Reinitialise the matcher, using the already computed chromagram.
        '''         
        midi_obj = self.matcher.midi_obj
        
        # Re-initialise the matcher, making sure that we have the same args as the 
        # ones used for initialisation.
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length,
                               compute_chromagram_fcn=self.matcher.compute_chromagram_fcn,
                               compute_chromagram_fcn_kwargs=self.matcher.compute_chromagram_fcn_kwargs, 
                               chromagram_act=self.matcher.dtw.features_act)
        
        self.matcher.midi_obj = midi_obj 
        self.status = STOP
        
        # Clear the audio history
        self.audio_data = np.array([])
        
        # Re-initialise the music detection diagnostic
        self.music_detection_diagnostic = ''
        self.music_detected_all = []   
 
    def bind_sockets(self):
        '''
        Start the ZMQ sockets
        '''
        context = zmq.Context()
        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.bind("tcp://127.0.0.1:5555")
        
        self.socket_sub = context.socket(zmq.SUB)                
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, r'') # Need to use different options for python3 vs python2                    
        self.socket_sub.connect("tcp://127.0.0.1:5556")
        
    
        self.socket_req_init = context.socket(zmq.REQ)
        self.socket_req_init.connect("tcp://127.0.0.1:5557")                        
        
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
        
    def update_status_matcher(self):
        '''
        Check if we should start the matching procedure.
        The function ensures that music has been detected.        
        '''
         
        # Run the music detection procedure on the most recent samples   
        if self.detect_music: 
            audio_data_tmp = self.audio_data[max(len(self.audio_data)-self.music_detecter.nb_sample, 0):len(self.audio_data)]                       
            music_detected, self.music_detection_diagnostic = self.music_detecter.detect(audio_data_tmp)
            
            self.music_detected_all.append(music_detected)
                    
            if music_detected and self.status_matcher == PAUSE:
                self.status_matcher = MATCH
            elif not music_detected and self.status_matcher == MATCH:
                self.status_matcher = PAUSE
        else:
            self.status_matcher = MATCH                                
    
    def callback_main(self, new_audio_data):
        '''
        The callback, called for every CHUNK of new data.
        Non-blocking callback.
        Process CHUNK samples of data, updating the matcher.
        
        ''' 
        # Start by converting the input string to numpy array
        new_audio_data = np.fromstring(new_audio_data, dtype=utils.AUDIO_FORMAT_MAP[self.audio_format][0])
        
        # Keep the entire history of audio_data (not required, we could only keep the last few samples)        
        self.audio_data = np.append(self.audio_data, new_audio_data)        
        
        # Check the status
        self.update_status_matcher()
        if self.status_matcher == PAUSE:
            return    
        
        # Run the online matching with the new data only.
        # This will be a problem if we use a larger music detection buffer, careful!!!
        self.matcher.match_online(new_audio_data)
            
    def publish_position(self):
        '''
        Send the current position to the GUI via the socket.
        '''
        # Disable the logging for now, as it causes delays on the front-end side
        root_str = "Sending current position: {} ({:.1f}s)         Status: {}({})".format(
            self.matcher.position_tick,
            self.matcher.positions_sec[-1] if len(self.matcher.positions_sec)>0 else 0.0,
            self.status_matcher, 
            self.music_detection_diagnostic)
                        
        print(root_str)
        
        # Need to be careful when sending the bytes representation of an int
        # The bytes functions behaves differently in python 2 vs python 3.                   
        # Send the position in millisec (alternatively, use self.matcher.position_tick for the midi ticks).
        position_millisec = int(self.matcher.positions_sec[-1]*1000.0) if len(self.matcher.positions_sec)>0 else 0
        self.socket_pub.send_string(str(position_millisec))
                            
    def save(self):
        '''
        Store the MatcherManager
        '''
        from pickle import dump
        
        filename_output = utils.WD + 'ScoreFollowingLogs\matcher.pkl' 
        file_output = open(filename_output, 'wb')
        dump(self, file_output)        
        
        print('MatcherManager stored in {}'.format(filename_output))
            
            
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
        # Receive the instruction from the front-end
        try:
            flag = self.socket_sub.recv(flags=zmq.NOBLOCK) # What happens when we have multiple messages?? Maybe set the SUB to CONFLATE                                           
        except zmq.Again:
            flag = self.status
            
        # Check if we have attained the end of the act features
        reached_end_act = len(self.matcher.positions) > 0 and self.matcher.positions[-1] >= self.matcher.nb_frames_feature_act
        flag = STOP if reached_end_act else flag  
                        
        if self.status == STOP and flag == STOP:
            pass
        
        elif (self.status == START and flag == START) or (self.status == STOP and flag == START): 
            self.callback_main(new_audio_data)
            self.status = START
            
        elif self.status == START and flag == STOP:
            # Send a stop instruction, save and reinitialise
            self.socket_pub.send(bytes(STOP))
            self.save()            
            self.reinit()            
            self.music_detection_diagnostic = ''            
            
        else:
            raise(ValueError('Unexpected status ({}) or flag ({})'.format(self.status, flag)))                    
                    
            
def callback(in_data, frame_count, time_info, flag):
    '''
    We need to have a callback outside the class.
    This function is called in a  new thread.
    It could be a problem if there are high fixed cost to copy the matcher in the new thread.    
    '''
    global matcher_manager
    matcher_manager.callback_fcn(matcher_manager, in_data)
    return (None, pyaudio.paContinue)
            
if __name__ == '__main__':    
    
    # Input the target '.mid' file of the score that we want to follow 
    if len(sys.argv) > 1:
        filename_mid = sys.argv[1]
        callback_fcn = MatcherManager.callback        
    else:
        filename_mid = r'C:\Users\Alexis\Source\seescore\xml samples/35882-Fur_Elise_by_Ludwig_Van_Beethoven.mid'
        callback_fcn = MatcherManager.callback_main
     
    matcher_manager = MatcherManager(filename_mid, callback_fcn)
    matcher_manager.socket_req_init.send(INIT)        
     
    while matcher_manager.stream.is_active():
        matcher_manager.publish_position()
#         sleep(matcher_manager.chunk/float(matcher_manager.sr))
        sleep(0.5)
                 
    matcher_manager.stream.stop_stream()
    matcher_manager.stream.close()    