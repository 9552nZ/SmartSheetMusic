import pyaudio
import numpy as np

class MicrophoneStreamer(object):
    def __init__(self, samplerate=44100, inchannels=1, chunk = 4410, buffersize=100):
        self.frames = []
        self.times = []
        self.channels = inchannels
        self.p = pyaudio.PyAudio()
        self.buffersize = buffersize
       
        self.stream = self.p.open(format = pyaudio.paFloat32,
                                  channels = inchannels, 
                                  rate = samplerate, 
                                  output = False,
                                  input = True, 
                                  frames_per_buffer = chunk,
                                  stream_callback = self.callback)      
        self.stream.stop_stream()

    def callback(self, data, frame_count, time_info, flag):
        audio_data = np.fromstring(data, dtype=np.float32) #.reshape(-1, self.channels)        
        self.frames.append(audio_data)  
        self.times.append(time_info['current_time'])
        if len(self.times)>self.buffersize:
            self.frames.pop(0)
            self.times.pop(0)
        return None, pyaudio.paContinue
    
    def get_frames(self):
            frames = self.frames
            times = self.times        
            return frames, times
            
    def start(self):
        try:
            self.stream.start_stream()
        except KeyboardInterrupt:
            self.stream.stop_stream()        

    def stop(self):
        self.stream.stop_stream()              

    def close(self):
        self.stream.close()
        self.p.terminate()
        
