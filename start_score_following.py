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
from time import sleep
import pyaudio
import numpy as np
import sys
import zmq
import utils_audio_transcript as utils
from score_following import Matcher
from music_detection import MusicDetecter
from pandas import DataFrame
import os # REMOVE
import datetime # REMOVE
from multiprocessing import Process, Manager, Pool
from _overlapped import NULL

# Flag for the communication with the GUI
# TODO : Replace by integers
STOP = b"stop"
START = b"start"
MATCH = b"match"
PAUSE = b"pause"
INIT = b"init"

# Start the ZMQ sockets
context = zmq.Context()
SOCKET_PUB = context.socket(zmq.PUB)
PUB_ADDRESS = r"tcp://127.0.0.1:5555"
SOCKET_SUB = context.socket(zmq.SUB)
SOCKET_SUB.setsockopt_string(zmq.SUBSCRIBE, r'')                
SOCKET_SUB.connect("tcp://127.0.0.1:5556")

# The frequency at which we publish (in secs)
PUB_FREQ = 0.1 

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
        self.detect_music = False # Set to true to only start once the MLP has detected music
        self.callback_fcn = callback_fcn # The pyaudio callback function
         
        # Split the path of the midi input between the directory and file name.
        (wd, self.filename) = os.path.split(filename_mid_full)
        self.wd = wd + "/"      
        
        # We initialise with a INIT status for the GUI and a PAUSE for the matcher         
        self.status = INIT
        self.status_matcher = PAUSE   
        self.music_detection_diagnostic = ''                      
        
    def __getstate__(self):            
        '''
        Called by pickle.dump to know what attributes need to be pickled. 
        Pickle everything but the sockets and pyaudio.
        '''
        return {k: v for k, v in self.__dict__.items() if k not in ['socket_pub', 'socket_sub', 'pyaudio', 'stream', 'callback_fcn']}
    
    def warm_up(self, status_init):
        '''
        Do all the expensive part of the initialisation.
        status_init is a shared Value() between two processes.
        '''
        
        # Initialise the matcher
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length, 
                               compute_chromagram_fcn_kwargs={'n_fft':self.hop_length*2, 'n_chroma':12})
        
        # Initialise the audio data buffer
        self.audio_data = np.array([])
     
        # Initialise the music detecter        
        if self.detect_music:
            self.music_detecter = MusicDetecter(utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\", self.sr)    
            
        # Set the status to STOP
        status_init['value'] = STOP
        
    def reinit(self):
        '''
        Reinitialise the audio data.
        Reinitialise the matcher, using the already computed chromagram.
        '''         
        midi_obj = self.matcher.midi_obj
        
        # Reinitialise the matcher, making sure that we have the same args as the 
        # ones used for initialisation.
        self.matcher = Matcher(self.wd, self.filename, self.sr, self.hop_length,
                               compute_chromagram_fcn=self.matcher.compute_chromagram_fcn,
                               compute_chromagram_fcn_kwargs=self.matcher.compute_chromagram_fcn_kwargs, 
                               chromagram_act=self.matcher.dtw.features_act)
        
        self.matcher.midi_obj = midi_obj 
        self.status = STOP
        
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
            
        if self.detect_music:                        
            music_detected, self.music_detection_diagnostic = self.music_detecter.detect(self.audio_data)
                    
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
        
        # Keep a local buffer (we keep as many samples as the music detection procedure requires)
        # TODO : move to update_status_matcher()
        if self.detect_music:
            self.audio_data = np.append(self.audio_data, new_audio_data)        
            self.audio_data = self.audio_data[max(len(self.audio_data)-self.music_detecter.model.nb_sample, 0):len(self.audio_data)]
        
        # Check the status
        self.update_status_matcher()
        if self.status_matcher == PAUSE:
            return    
        
        # Run the online matching with the new data only.
        # This will be a problem if we use a larger music detection buffer, careful!!!
        self.matcher.match_online(new_audio_data)                    
                
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
            flag = SOCKET_SUB.recv(flags=zmq.NOBLOCK) # What happens when we have multiple messages?? Maybe set the SUB to CONFLATE                                           
        except zmq.Again:
            flag = self.status
            
        # Check if we have attained the end of the act features
        reached_end_act = len(self.matcher.positions) > 0 and self.matcher.positions[-1] >= self.matcher.nb_obs_feature_act
        flag = STOP if reached_end_act else flag  
                        
        # If we have not yet received any instruction, or if the matcher is not initialised do nothing                        
        if (self.status == STOP and flag == STOP) or (self.status == INIT):
            pass
        
        elif (self.status == START and flag == START) or (self.status == STOP and flag == START): 
            self.callback_main(new_audio_data)
            self.status = START
            
        elif self.status == START and flag == STOP:
            # Send a stop instruction, save and reinitialise
            publish(bytes(STOP)) # BYTES OR STR?????
            self.save()            
            self.reinit()            
            self.music_detection_diagnostic = ''            
            
        else:
            raise(ValueError('Unexpected status ({}) or flag ({})'.format(self.status, flag)))
        
def check_warm_up_status(status):
    '''
    While the manager is not initialised, publish the INIT flag.
    The status variable is shared between the two processes.
    '''        
    SOCKET_PUB.bind(PUB_ADDRESS)
    while status['value'] == INIT:        
        publish(INIT)
        sleep(PUB_FREQ)
            
        SOCKET_PUB.unbind(PUB_ADDRESS)
            
def publish(data):
    '''
    Send instructions to the front end via the socket.
    '''        
    root_str = "Sending: {}"
    print(root_str.format(str(data)))   # BYTES OR STR?????
    
    # Need to be careful when sending the bytes representation of an int
    # The bytes functions behaves differently in python 2 vs python 3.            
    SOCKET_PUB.send_string(str(data))
                                
            
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
        filename_mid = utils.WD + '\Samples\IMSLP//' +  'Beethoven_Moonlight_Sonata_Op._27_No._2_Mvt._2.wav'
        callback_fcn = MatcherManager.callback_main
 
    # Initialise the MatcherManager (fast) 
    matcher_manager = MatcherManager(filename_mid, callback_fcn)
    
    context2 = zmq.Context()
    socket = context2.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5557")
    socket.send(b"Hello")
    message = socket.recv()        
    
    # We need to update the front-end about the warming up (slow)
    # Start an ancillary process to monitor the progress of the 
    # warming up.
    status = Manager().dict()
    status['value'] = INIT                                     
    process_check_warm_up = Process(target=check_warm_up_status, args=(status, ))
    process_check_warm_up.start()    
    
    matcher_manager.warm_up(status)
    
    # Once the warm-up is done, we can bind the socket in the main thread
    while process_check_warm_up.is_alive():
        sleep(PUB_FREQ)
    
    SOCKET_PUB.bind(PUB_ADDRESS)    

    # Start the pyaudio stream 
    matcher_manager.start_stream()    
     
    while matcher_manager.stream.is_active():
        publish(matcher_manager.matcher.position_tick)
        sleep(PUB_FREQ)
                 
    matcher_manager.stream.stop_stream()
    matcher_manager.stream.close()    
    SOCKET_PUB.unbind(PUB_ADDRESS)