# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:34:42 2017

@author: Niki
"""
import pyaudio
import numpy as np
import librosa
import pretty_midi
import MicrophoneStreamer
import time

class ScoreFollow(object):
    
    def __init__(self):
        self.samplerate = 44100
        self.stepseconds = 0.2 # in seconds
        self.stepsinupdate = 10
        self.stepsinwindow = 50
        self.chunk = 4410
        self.channels = 1
        self.instrument = 0 # Grand Piano
        self.scorefile = ''
        self.midifile = r'progression.mid';
        self.soundfont = br"C:\Users\Niki\Datasets\Soundfonts\Weeds\WeedsGM3.sf2";
        
        self.mic = MicrophoneStreamer.MicrophoneStreamer(samplerate=44100, inchannels = self.channels, chunk = self.chunk)

        self.pa = pyaudio.PyAudio()
        self.streamOut = self.pa.open(
                format = pyaudio.paFloat32,
                channels = self.channels, 
                rate = self.samplerate, 
                output = True,
                input = False, 
                )

    def loadScore(self, scoreFile):
        return 0

    def score2midi(self, scoreObj):
        return 0
    
    def loadMidi(self):
        midiObj = pretty_midi.PrettyMIDI(self.midifile)
        for instrument in midiObj.instruments:        
            instrument.program = self.instrument
        return midiObj
        
    def midi2wav(self, midiObj):
        wavObj = midiObj.fluidsynth(fs=self.samplerate, sf2_path = self.soundfont)
        return wavObj
    
    def playWav(self, wavObj):
        data = wavObj.astype(np.float32).tostring()
        print('Starting playback')
        self.streamOut.write(data)

    def wav2features(self, wavObj):
        idx = 0
        step = int(self.samplerate * self.stepseconds)
        while idx<len(wavObj):
            subwav = wavObj[idx:idx+step]
            #f = librosa.feature.mfcc(subwav, n_mfcc=5)
            f = librosa.feature.chroma_cqt(y=subwav,sr=self.samplerate)
            f = f.reshape(f.shape[0]*f.shape[1])
            f /= np.std(f, axis=0)
            if idx==0:
                features = f
            elif features.shape[0] == f.shape[0]:
                features = np.column_stack((features, f))
            idx += step
        return features
    
    def mic2features(self):
        if (not self.mic.frames) and (not self.mic.stream.is_active()):
            print('Microphone is not active!')
            return None
        frames, times = self.mic.get_frames()
        numfeat = int(self.stepseconds*self.samplerate/self.chunk)
        ranges = np.linspace(len(frames)-self.stepsinupdate*numfeat, len(frames), self.stepsinupdate+1).astype(int)
        for i in range(self.stepsinupdate):
            subwav = np.concatenate(frames[ranges[i]:ranges[i+1]])
            #feature = librosa.feature.mfcc(subwav, n_mfcc=5)
            f = librosa.feature.chroma_cqt(y=subwav,sr=self.samplerate)
            f = f.reshape(f.shape[0]*f.shape[1])
            f /= np.std(f, axis=0)
            if i==0:
                features = f
            elif features.shape[0] == f.shape[0]:
                features = np.column_stack((features, f))
        return features
        
    def feature2distance(self, livefeature, features):
        if livefeature.ndim==1:
            livefeature.resize(len(livefeature),1)
        dist = np.zeros([features.shape[1], livefeature.shape[1]])
        for i in range(livefeature.shape[1]):
            dist[:, [i]] = np.mean((features - livefeature[:,[i]])**2, axis=0, keepdims=True).T
        return dist
    
    def probModel(self, dist, dt, timewindow):      
        p1 = np.exp(-dist);
        p2 = np.exp(-abs(timewindow-dt)/3)
        
        p = p1*p2
        #p = p1
        p /= np.sum(p, axis=0)
        return p
    
    def startFollowing(self, features):
        if not self.mic.stream.is_active():
            self.mic.start()
        print("Starting score following. Exit with 'Ctrl+C'.")
        if len(self.mic.frames)<100:
            time.sleep(2)
        windowsize = self.stepseconds * self.stepsinwindow # in seconds
        timewindow = np.linspace(-windowsize, windowsize, 2*self.stepsinwindow+1)
        timewindow.resize((len(timewindow),1))
        nfeat      = np.size(features,1)
            
        currentpos = 0
        noiselevel = 0
        speed = 0
        t0 = time.clock()
        try:
            while True:
                time.sleep(0.1)
                livefeatures  = self.mic2features()
                
                featstart = max([0, currentpos-self.stepsinwindow])
                featend = min([nfeat, currentpos+self.stepsinwindow])
                thisfeatures = features[:,featstart:featend]
                
                dist = self.feature2distance(livefeatures, thisfeatures)
        
                timestart = max(0, self.stepsinwindow-currentpos)
                timeend = min(2*self.stepsinwindow, self.stepsinwindow+nfeat-currentpos)
                currentwindow =  timewindow[timestart:timeend]
                avprob = 1/(len(currentwindow))
                
                dt = time.clock() - t0
                t0 = time.clock()
                p = self.probModel(dist, dt, currentwindow)
                
                idx = p.argmax(axis=0)
                val = p.max(axis=0)
                index = np.sum(idx*val)/np.sum(val)
                avval = np.mean(val)
                
                posDelta = index - self.stepsinwindow + timestart
                oldpos   = currentpos
                noiselevel = min(0.6*noiselevel + 0.5*np.mean(livefeatures), 2.5)
                if noiselevel>2: # this is noise - reset
                    currentpos = 0
                    speed = 0
                elif avval>3*avprob: # we've found a match, update position
                    speed = 0.6*speed + 0.4*posDelta            
                    currentpos += speed
                    currentpos = int(currentpos)
                
                if oldpos!=currentpos:
                    print('Current Frame:', currentpos, avval, avprob)
                
        except KeyboardInterrupt:
            self.mic.stop()
