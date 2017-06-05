import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import utils_audio_transcript as utils

class Matcher():
    
    def __init__(self, chromagram_act):
        
        # Current position in the act / est data
        # Set to -1 as no data has been processed
        self.idx_act = -1
        self.idx_est = -1
        
        # Store the target chromagram 
        self.chromagram_act = chromagram_act
        
        # Online DTW parameters
        self.min_data = 5 # min number of input frames to compute before doing anything
        self.search_width = 100 # width of the band to search the optimal alignment (in frames) 
        
        # Boolean to check if the new input has been processed
        self.input_advanced = False
        
        # Initialise large empty matrices for chromagram_est and for the cumulative distance matrix
        self.chromagram_est = np.zeros((chromagram_act.shape[0]*3, chromagram_act.shape[1]))
        self.chromagram_est.fill(np.nan)
        self.cum_distance = np.zeros((chromagram_act.shape[0], chromagram_act.shape[0]*3))
        self.cum_distance.fill(np.nan)
         
        
    def select_advance_direction(self):        
        
        idx_est = self.idx_est
        idx_act = self.idx_act
            
        if idx_est <= self.min_data or idx_act <= self.min_data:
            return(0)  
        
        # Check if the minimum distance is in the last row or in the last column
        arg_min_row = np.nanargmin(self.cum_distance[idx_act,0:idx_est+1])
        arg_min_col = np.nanargmin(self.cum_distance[0:idx_act+1,idx_est])
                
        if arg_min_row == idx_est and arg_min_col == idx_act:
            direction = 0 # compute both row and column
        elif self.cum_distance[idx_act,arg_min_row] < self.cum_distance[arg_min_col,idx_est]:
            direction = 1 # compute next row
        else:
            direction = 2 # compute next column  
                        
        return(direction)

    def find_cells(self, i, j):
        '''
        The function identifies the cells that should be evaluated.
        Refer to Figure 2 of http://eecs.qmul.ac.uk/~simond/pub/2005/dafx05.pdf
        for detail on the possible cases        
         
        '''
        
        if i-1 < 0:
            return(([i],[j-1]))            
            
        if j-1 < 0:
            return(([i-1],[j]))
        
        isnan_s = np.isnan(self.cum_distance[i-1, j])
        isnan_w = np.isnan(self.cum_distance[i, j-1])
        isnan_sw = np.isnan(self.cum_distance[i, j-1])
        
        # Standard case: compute everything
        # The case with all 3 nans should not arise
        if not isnan_s and not isnan_w and not isnan_sw:
            return(([i-1, i, i-1],[j, j-1, j-1]))
        
        elif isnan_s and isnan_sw:
            return(([i],[j-1]))
        
        elif isnan_w and isnan_sw:
            return(([i-1],[j]))
        
        elif isnan_s:
            return(([i,i-1], [j-1, j-1]))
        
        elif isnan_w:
            return(([i-1, i-1],[j, j-1]))
        
    def find_best_path_distance(self, cells):
        '''
        The function computes the local DTW path, adjusting for the weigths.
        It relies on the cells being passed such that the diagonal is the 
        last item (if two or three elements).
        '''
        len_cells = len(cells[0])        
                    
        if len_cells == 3:
            return(min(self.cum_distance[cells[0][0], cells[1][0]], 
                       self.cum_distance[cells[0][1], cells[1][1]], 
                       2.0 * self.cum_distance[cells[0][2], cells[1][2]]))
        
        if len_cells == 2:
            return(min(self.cum_distance[cells[0][0], cells[1][0]], 2.0 * self.cum_distance[cells[0][1], cells[1][1]]))
        
        if len_cells == 1:
            return(self.cum_distance[cells[0][0], cells[1][0]])
        
         
    def update_cum_distance(self, direction):
        '''
        Taking the existing cum_distance matrix and the search direction, this function
        updates the cum_distance by finding the (local) min distance path.
        '''
        
        # For the first iteration, just take the distance between the two chromagrams
        if self.idx_act == -1 and self.idx_est == -1:
            self.idx_act += 1
            self.idx_est += 1
            self.cum_distance[self.idx_act, self.idx_est] = utils.distance_chroma(chromagram_act[self.idx_act, :], 
                                                                                  self.chromagram_est[self.idx_est, :], 1, 12)
            self.input_advanced = True            
            return                        
        
        if direction == 0:
            # Update in both directions, but start by updating in the live feed direction
            # (not required, but this is what Dixon does)
            self.update_cum_distance(2)
            self.update_cum_distance(1)
            return
                        
        if direction == 1:
            # Advance the score
            self.idx_act += 1
            idx_act = self.idx_act
            idx_est = self.idx_est
            
            for k in np.arange(max(idx_est-self.search_width-1, 0), idx_est+1):                
                distance = utils.distance_chroma(chromagram_act[idx_act, :], self.chromagram_est[k, :], 1, 12)                
                cells = self.find_cells(idx_act, k)
                self.cum_distance[idx_act, k] = distance + self.find_best_path_distance(cells)                                                                 
            return
            
        if direction == 2:
            # Advance the live feed
            self.idx_est += 1
            self.input_advanced = True  
            idx_act = self.idx_act
            idx_est = self.idx_est
            
            for k in np.arange(max(idx_act-self.search_width-1, 0), idx_act+1):
                distance = utils.distance_chroma(chromagram_act[k, :], self.chromagram_est[idx_est, :], 1, 12)
                cells = self.find_cells(k, idx_est)
                self.cum_distance[k, idx_est] = distance + self.find_best_path_distance(cells)
            return
            
    def update_best_path(self):
        '''
        Based on the cum distance matrix, this function finds the best path
        Running this function is not required for the main loop, it serves mainly 
        for ex-post analysis.
        ''' 

        idx_est = self.idx_est
        idx_act = self.idx_act
        
        # Check if the minimum distance is in the last row or in the last column
        arg_min_row = np.nanargmin(self.cum_distance[idx_act,0:idx_est+1])
        arg_min_col = np.nanargmin(self.cum_distance[0:idx_act+1,idx_est])
        
        path_rows = []
        path_cols = []
        
        # Collect the final point of the DTW path
        if self.cum_distance[idx_act,arg_min_row] <= self.cum_distance[arg_min_col,idx_est]:
            path_rows.append(idx_act)
            path_cols.append(arg_min_row)
            idx_est = arg_min_row
        else:
            path_rows.append(arg_min_col)
            path_cols.append(idx_est)
            idx_act = arg_min_col
            
        # Iterate until the starting point
        while idx_act > 0 or idx_est > 0:
            cells = self.find_cells(idx_act, idx_est)
        
            
                                                        
    def main_matcher(self, chromagram_est_row): 
        '''
        Called for every new frame of the live feed, for which we run the online DTW
        '''
        # Disable the matching procedure if we are at the end of the act data
        if self.idx_act >= self.chromagram_act.shape[0] - 1:
            return
        
        self.chromagram_est[self.idx_est+1, :] = chromagram_est_row
        self.input_advanced = False

        while not self.input_advanced:
            direction = self.select_advance_direction()
            self.update_cum_distance(direction)                
        
        self.update_best_path()

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
            
wd = "C:/Users/Alexis/Business/SmartSheetMusic/Samples/Nocturnes/"
filename_act = wd + "nocturnes.wav"
# filename_est = wd + "Yundi Li plays Chopin Nocturne Op. 9 No. 2.wav"
filename_est = filename_act

SR = 11025
N_FFT = 2048
HOP_LENGTH = 1024

audio_data_act = lb.core.load(filename_act, sr = SR, offset=0.0, duration=None)[0]
audio_data_est = lb.core.load(filename_est, sr = SR, offset=0.0, duration=None)[0]

win_len_smooth = 1
tuning = 0.0
# chromagram_act = lb.feature.chroma_cens(y=audio_data_act, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T
# chromagram_est = lb.feature.chroma_cens(y=audio_data_est, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T
# np.save( wd + 'chromagram_act.npy', chromagram_act)
# np.save( wd + 'chromagram_est.npy', chromagram_est)
chromagram_act = np.load( wd + 'chromagram_act.npy')
chromagram_est = np.load( wd + 'chromagram_est.npy')

nb_frames_est = chromagram_est.shape[0]

matcher = Matcher(chromagram_act) 

for n in range(nb_frames_est):
# for n in range(500):
    print n
    matcher.main_matcher(chromagram_est[n,:]) 
    
a = 1

data = matcher.cum_distance[0:500, 0:500]
data = matcher.cum_distance[0:1000, 0:1000]
data = matcher.cum_distance[1000:2000, 1000:2000]
plt.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, vmin=np.nanmin(data), vmax=np.nanmax(data))
    




