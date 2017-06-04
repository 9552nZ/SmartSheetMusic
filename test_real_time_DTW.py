import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import utils_audio_transcript as utils

class Matcher():
    
    def __init__(self, chromagram_act):
        self.idx_act = -1
        self.idx_est = -1
        self.chromagram_act = chromagram_act
        
        # Boolean to check if the new input has been processed
        self.input_advanced = False
        
        # Initialise large empty matrices for chromagram_est and for the cumulative distance matrix
        self.chromagram_est = np.empty((chromagram_act.shape[0]*3, chromagram_act.shape[1]))
        self.cum_distance = np.empty((chromagram_act.shape[0], chromagram_act.shape[0]*3))
         
        
    def select_advance_direction(self):
    
        nb_init = 50 #min number of rows or columns to compute before doing anything
        
        idx_est = self.idx_est # number of columns in the cum_distance matrix
        idx_act = self.idx_act
            
        if idx_est <= nb_init or idx_act <= nb_init:
            return(0)  
        
        arg_min_row = np.argmin(self.cum_distance[idx_act,:])
        arg_min_col = np.argmin(self.cum_distance[:,idx_est])
                
        if self.cum_distance[idx_act,arg_min_row] == self.cum_distance[arg_min_col,idx_est]:
            direction = 0 # compute both row and column
        elif self.cum_distance[idx_act,arg_min_row] < self.cum_distance[arg_min_col,idx_est]:
            direction = 1 # compute next row
        else:
            direction = 2 # compute next column  
                        
        return(direction)
    
    def update_cum_distance(self, direction):
        
        def best_path(south, west, south_west):                                
            return(min(south, west, 2*south_west))
        
        # For the first iteration, just take the distance between the two chromagrams
        if self.idx_act == -1 and self.idx_est == -1:
            self.idx_act += 1
            self.idx_est += 1
            self.cum_distance[self.idx_act, self.idx_est] = utils.distance_chroma(chromagram_act[self.idx_act, :], 
                                                                                  self.chromagram_est[self.idx_est, :], 1, 12)
            self.input_advanced = True
            
            return
                        
        c = 500
        
        if direction == 0:
            self.update_cum_distance(1)
            self.update_cum_distance(2)
            
        if direction == 1:
            # Advance the score
            self.idx_act += 1
            idx_act = self.idx_act
            idx_est = self.idx_est
            
            for k in np.arange(max(idx_est-c-1, 0), idx_est+1):
                distance = utils.distance_chroma(chromagram_act[idx_act, :], self.chromagram_est[k, :], 1, 12)
                self.cum_distance[idx_act, k] = distance + best_path(self.cum_distance[idx_act-1, k],
                                                                     self.cum_distance[idx_act, k-1],
                                                                     self.cum_distance[idx_act-1, k-1])
                                                      
                                                        
        if direction == 2:
            # Advance the live feed
            self.idx_est += 1
            self.input_advanced = True  
            idx_act = self.idx_act
            idx_est = self.idx_est
            
            for k in np.arange(max(idx_act-c-1, 0), idx_act+1):
                distance = utils.distance_chroma(chromagram_act[k, :], self.chromagram_est[idx_est, :], 1, 12)
                self.cum_distance[k, idx_est] = distance + best_path(self.cum_distance[k-1, idx_est],
                                                                     self.cum_distance[k, idx_est-1],
                                                                     self.cum_distance[k-1, idx_est-1])
                                                        
    def main_matcher(self, chromagram_est_row): 
        '''
        Called for every new frame of the live feed, for which we run the online DTW
        '''
        
        self.chromagram_est[self.idx_est+1, :] = chromagram_est_row
        self.input_advanced = False

        while not self.input_advanced:
            direction = self.select_advance_direction()
            self.update_cum_distance(direction)                

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
            
wd = "C:/Users/Alexis/Business/SmartSheetMusic/Samples/Nocturnes/"
filename_act = wd + "nocturnes.wav"
filename_est = wd + "Yundi Li plays Chopin Nocturne Op. 9 No. 2.wav"

SR = 11025
N_FFT = 2048
HOP_LENGTH = 1024

audio_data_act = lb.core.load(filename_act, sr = SR, offset=0.0, duration=None)[0]
audio_data_est = lb.core.load(filename_est, sr = SR, offset=0.0, duration=None)[0]

win_len_smooth = 1
tuning = 0.0
chromagram_act = lb.feature.chroma_cens(y=audio_data_act, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T
chromagram_est = lb.feature.chroma_cens(y=audio_data_est, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T

# nb_frames_act = chromagram_act.shape[0]
nb_frames_est = chromagram_est.shape[0]

matcher = Matcher(chromagram_act) 

for n in range(nb_frames_est):
    matcher.main_matcher(chromagram_est[n,:]) 
    




