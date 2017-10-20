# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:34:42 2017

@author: Niki
"""
import numpy as np
import librosa
import pretty_midi
import sounddevice as sd
import time
import statistics
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class ScoreFollow(object):
    
    def __init__(self):
        self.samplerate = 11025
        self.hop_size = 256
        self.windowsize = 3  # in seconds
        self.livewindow = 3 # in seconds
        self.buffersize = 5 # in seconds   
        self.framesinbuffer = self.samplerate*self.buffersize
        
        self.frames = np.zeros((0,1))
        
        self.reading_prob = [0.5]
        self.reading_speed = [40]
        self.reading_pos = [88]
        self.volume = []
        self.kfilter = self.kalman_initialise()
        self.kalman_out = []
        self.speed = []

        self.debug_features = []
        self.debug_livefeatures = []
        self.debug_livewav = []
        self.debug_dist = []
        
        self.channels = 1
        self.instrument = 0 # Grand Piano
        self.scorefile = ''
        self.midifile = r'progression.mid';
        self.soundfont = br"C:\Users\Niki\Datasets\Soundfonts\Weeds\WeedsGM3.sf2"
        
        self.mic = sd.InputStream(channels=self.channels, 
                       callback=self.callback,
                       samplerate=self.samplerate)
        
    def callback(self, indata, frames, time, status):  
        self.frames = np.concatenate((self.frames, indata))
        if self.frames.shape[0]>self.framesinbuffer:
            self.frames = self.frames[-self.framesinbuffer-1:-1]
            

    def load_score(self, scoreFile):
        return 0

    def score2midi(self, scoreObj):
        return 0
    
    def load_midi(self):
        midiObj = pretty_midi.PrettyMIDI(self.midifile)
        for instrument in midiObj.instruments:        
            instrument.program = self.instrument
        return midiObj
        
    def midi2wav(self, midiObj):
        wavObj = midiObj.fluidsynth(fs=self.samplerate, sf2_path = self.soundfont)
        return wavObj
    
    def play(self, wavObj):
        sd.play(wavObj, self.samplerate)

    def stop(self):
        sd.stop()

    def wav2features(self, wavObj):
        features = librosa.feature.chroma_cqt(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, norm = 2, threshold=0.001, n_chroma=7*12, n_octaves=7)
        return features
    
    def feature2distance(self, livefeature, features):
        if livefeature.ndim==1:
            livefeature.resize(len(livefeature),1)
        dist = np.zeros([features.shape[1], livefeature.shape[1]])
        for i in range(livefeature.shape[1]):
            dist[:, [i]] = np.mean((features - livefeature[:,[i]])**2, axis=0, keepdims=True).T / np.mean(features**2, axis=0, keepdims=True).T
        return dist
    
    def prob_model(self, dt, dist, featuretime, speed=43):
        [ntrue, nlive] = dist.shape
        for j in range(0,40):
            offset = featuretime*(speed+j-20)
            p1 = np.exp(-dist)
            for i in range(nlive-1):
                n = int((nlive-i)*offset)
                if n>0:
                    p1[n:,[i]] = p1[0:-n,[i]]
                    p1[0:n,[i]] = 0.1            
                p = np.prod(p1,axis=1)
                p /= np.sum(p)
            if j==0:
                probs = p
            else:
                probs = np.concatenate([probs,p])
                
        order = probs.argsort()
        val = probs[order[-2:]]
        spidx = np.sum(val*np.floor(order[-2:]/ntrue))/np.sum(val)
        deltaidx = np.sum(val*np.mod(order[-2:], ntrue))/np.sum(val)
        
        self.reading_pos.append(self.reading_pos[-1]+deltaidx-ntrue/2)
        self.reading_speed.append(np.mean(speed+spidx-20))
        self.reading_prob.append(np.mean(val))
             
    def kalman_initialise(self, dim_x = 3, dim_z = 2, dt = 0.5):
        kfilter = KalmanFilter(dim_x, dim_z)
        kfilter.x = np.array([88., 40., 0.])
        kfilter.F = np.array([[1, dt, 0.5*dt*dt],
                              [0,  1,        dt],
                              [0,  0,         1]], dtype=float)
        kfilter.H = np.array([[1, 0, 0],
                              [0, 1, 0]], dtype=float)
        kfilter.P = Q_discrete_white_noise(dim_x, 1, 1)
        kfilter.Q = 0.01*np.array([[0.1, 0, 0],
                              [0, 10, 0],
                              [0, 0, 100]], dtype=float)
        kfilter.R = np.array([[1, 0],
                              [0, 1]], dtype=float)    
        return kfilter
        
    def kalman_update(self, dt = 0.5, std_q = 1, std_r = 1):
        self.kfilter.F = np.array([[1, dt, 0.5*dt*dt],
                              [0,  1,        dt],
                              [0,  0,         1]], dtype=float)    
        
    def kalman_position_model(self):
        z = np.array([self.reading_pos[-1], self.reading_speed[-1]], dtype=float).squeeze()
        self.kfilter.update(z)
        self.kfilter.predict()       
        self.kalman_out.append(self.kfilter.x)
        
    def my_position_model(self):
        # to do
        return None 
        
    def start_following(self, features):       
        self.mic.start()
        time.sleep(self.livewindow)
        # One feature frame corresponds to 1024/44100 seconds
        featuretime = self.hop_size/self.samplerate
        defaultspeed = 1/featuretime
        numfeatures  = int(self.samplerate*self.windowsize/self.hop_size)
        currentpos   = numfeatures
        dt = 0
        print("Starting score following. Exit with 'Ctrl+C'.")
        try:
            while True:
                time.sleep(0.01)
                oldpos = currentpos
                if len(self.frames)<self.livewindow*self.samplerate:
                    livewav = self.frames.squeeze()
                else:
                    livewav = self.frames[-self.livewindow*self.samplerate:].squeeze()
                self.volume.append(np.sqrt(np.mean(livewav**2)))

                if self.volume[-1]>0.01:
                    thisfeatures = features[:,currentpos-numfeatures:currentpos+numfeatures]                    
                    livefeatures = self.wav2features(livewav)              
                    dist = self.feature2distance(livefeatures, thisfeatures)
                                     
                    if dt==0:
                        dt=0.6
                    else:
                        dt = time.clock() - t0
                    t0 = time.clock()
                    self.prob_model(dt, dist, featuretime, defaultspeed)
                    self.kalman_update(dt)
                    self.kalman_position_model()
                    
                    self.debug_features.append(thisfeatures)
                    self.debug_livefeatures.append(livefeatures)
                    self.debug_livewav.append(livewav)
                    self.debug_dist.append(dist)
                
                    self.speed.append(statistics.median(self.reading_speed[-10:]))

                    currentpos = max(int(self.kalman_out[-1][0]), numfeatures)
                                
                if oldpos!=currentpos:
                    print('Current Frame:', currentpos-numfeatures, 'Probability: ', self.reading_prob[-1], 'Speed:', self.speed[-1], 'dt:', dt)
                
        except KeyboardInterrupt:
            self.mic.stop()
            sd.stop()
