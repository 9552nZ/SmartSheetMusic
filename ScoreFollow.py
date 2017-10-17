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

class ScoreFollow(object):
    
    def __init__(self):
        self.samplerate = 44100
        self.hop_size = 1024
        self.windowsize = 2  # in seconds
        self.livewindow = 2 # in seconds
        self.buffersize = 5 # in seconds   
        self.framesinbuffer = self.samplerate*self.buffersize
        
        self.frames = np.zeros((0,1))
        self.speed = []
        self.speed2 = []

        self.debug_features = []
        self.debug_livefeatures = []
        self.debug_livewav = []
        self.debug_posdelta = []
        self.debug_dist = []
        self.debug_probs = []
        self.debug_currentpos = []
        self.debug_spid = []
        
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
        features = librosa.feature.chroma_cqt(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, norm = 2, threshold=0.0001, n_chroma=7*12, n_octaves=7)
        return features
    
    def feature2distance(self, livefeature, features):
        if livefeature.ndim==1:
            livefeature.resize(len(livefeature),1)
        dist = np.zeros([features.shape[1], livefeature.shape[1]])
        for i in range(livefeature.shape[1]):
            dist[:, [i]] = np.mean((features - livefeature[:,[i]])**2, axis=0, keepdims=True).T / np.mean(features**2, axis=0, keepdims=True).T
        return dist
    
    def prob_model_old(self, dist, featuretime, speed=43):      
        p1 = np.exp(-dist)
        nlive = p1.shape[1]
        offset = featuretime*speed
        for i in range(nlive-1):
            n = int((nlive-i)*offset)
            if n>0:
                p1[n:,[i]] = p1[0:-n,[i]]
                p1[0:n,[i]] = 0.1            
        p = np.prod(p1,axis=1)
        #p = np.mean(p1,axis=1)
        p /= np.sum(p)
        return p.argmax(), p.max(), 0
    
    def prob_model(self, dist, featuretime, speed=43):
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
        return deltaidx, np.mean(val), speed+spidx-20
                
    
    def start_following(self, features):       
        self.mic.start()
        time.sleep(self.livewindow)
        # One feature frame corresponds to 1024/44100 seconds
        featuretime = self.hop_size/self.samplerate
        numfeatures  = int(self.samplerate*self.windowsize/self.hop_size)
        currentpos   = numfeatures
        volume       = 0
        t0           = time.clock()
        print("Starting score following. Exit with 'Ctrl+C'.")
        try:
            while True:
                time.sleep(0.1)
                thisfeatures = features[:,currentpos-numfeatures:currentpos+numfeatures]
                
                if len(self.frames)<self.livewindow*self.samplerate:
                    livewav = self.frames.squeeze()
                else:
                    livewav = self.frames[-self.livewindow*self.samplerate:].squeeze()
                
                livefeatures = self.wav2features(livewav)              
                dist = self.feature2distance(livefeatures, thisfeatures)
                
                dt = time.clock() - t0
                t0 = time.clock()
                
                index, val, spid = self.prob_model(dist, featuretime,40)
                #print(index, val, spid, dt)
                posDelta = index-numfeatures
                if currentpos==numfeatures:
                    posDelta = max(0,posDelta)
                                               
                oldpos   = currentpos
                volume   = 0.6*volume+0.4*np.sqrt(np.mean(livewav**2))
                if volume<0.01: # this is noise - reset
                    currentpos = numfeatures
                elif val>0.2: # we've found a match, update position
                    # Loggind
                    self.debug_features.append(thisfeatures)
                    self.debug_livefeatures.append(livefeatures)
                    self.debug_livewav.append(livewav)
                    self.debug_posdelta.append(posDelta)
                    self.debug_dist.append(dist)
                    self.debug_currentpos.append(currentpos)
                    self.debug_spid.append(spid)
                    
                    self.speed.append(statistics.median(self.debug_posdelta[-10:])/dt)
                    self.speed2.append(statistics.median(self.debug_spid[-10:]))
                    currentpos += posDelta
                    currentpos = max(int(currentpos), numfeatures)

                                
                if oldpos!=currentpos:
                    print('Current Frame:', currentpos-numfeatures, 'Ratio: ', val, 'Speed:', self.speed2[-1])
                
        except KeyboardInterrupt:
            self.mic.stop()
            sd.stop()
