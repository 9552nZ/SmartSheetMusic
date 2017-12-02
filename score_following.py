import librosa as lb
import numpy as np
import os
import utils_audio_transcript as utils
import pretty_midi
import matplotlib.pyplot as plt
from math import ceil
        
class CellList():
    '''
    Class to handle list of cells.
    A cell is simply a tuple of coordinates (i,j) modeling the position in
    the cum_distance matrix.
    '''
    def __init__(self, rows, cols):
        
        self.len = len(rows)
        assert self.len == len(cols) 
        self.rows = rows
        self.cols = cols    
        
    def __repr__(self):
        return(np.array([np.array(self.rows).T, np.array(self.cols).T]).__repr__())   

    def append(self, cell):
        self.len += cell.len         
        self.rows.extend(cell.rows)
        self.cols.extend(cell.cols)
        
    def prepend(self, cell):
        self.len += cell.len
        self.rows = cell.rows + self.rows
        self.cols = cell.cols + self.cols 
        
    def get_cell(self, idx):        
        return(CellList([self.rows[idx]], [self.cols[idx]]))
    
    def get_cell_as_tuple(self, idx):
        return((self.rows[idx], self.cols[idx]))


class Matcher():
    '''
    Implementation of the online DTW matcher as per S. Dixon.
    Resources:
    http://eecs.qmul.ac.uk/~simond/pub/2005/dafx05.pdf
    https://www.eecs.qmul.ac.uk/~simond/pub/2010/Macrae-Dixon-ISMIR-2010-WindowedTimeWarping.pdf
    http://www.cp.jku.at/research/papers/Arzt_Masterarbeit_2007.pdf
    https://code.soundsoftware.ac.uk/projects/match/repository/entry/at/ofai/music/match/Finder.java
    '''
    def __init__(self, wd, filename, sr, hop_length,                  
                 diag_cost=1.20,
                 compute_chromagram_fcn = lb.feature.chroma_stft, # Change to use CQT                
                 compute_chromagram_fcn_kwargs = {}, 
                 chromagram_mode = 0,
                 chromagram_act = None,
                 use_low_memory=True): 
        
        # Start by fixing the sample rate and hop size for the spectrum
        # decomposition
        self.sr_act = sr
        self.hop_length_act = hop_length   
        
        # Set the function used to compute the chromagram and its parameters
        self.compute_chromagram_fcn = compute_chromagram_fcn
        self.compute_chromagram_fcn_kwargs = compute_chromagram_fcn_kwargs
        self.chromagram_mode = chromagram_mode
        
        # Check if we need to store the chromagram to disk and retrieve it when it exists
        self.store_chromagram = False  
        
        # Check whether we have the '.mid' or '.wav' as input.
        _, self.file_extension = os.path.splitext(filename)        
        if not (self.file_extension == '.mid' or self.file_extension == '.wav'):
            raise ValueError('Input file need to be either .mid or .wav')
        
        # Load the .wav file and turn into chromagram.
        self.set_chromagram(wd, filename, chromagram_act=chromagram_act)
                
        # Current position in the act / est data.
        # Set to -1 as no data has been processed.
        self.idx_act = -1
        self.idx_est = -1
        
        # Initialise position and position_sec and position_tick.  
        # They represent the expected position in the Midi file.
        self.position = [0]
        self.position_sec = [0]
        self.position_tick = 0
        
        # DTW parameters
        self.search_width = 100 # width of the band to search the optimal alignment (in frames)
        self.min_data = 20 # min number of input frames to compute before doing anything        
        self.diag_cost = diag_cost # the cost for the diagonal (1.0<=x<=2.0, may be set to less than 2 to let the algorithm favour diagonal moves) 
        
        # Boolean to check if the new input has been processed
        self.input_advanced = False
        
        # Initialise large empty matrices for chromagram_est and for the cumulative distance matrix
        self.chromagram_est = np.zeros((max(self.len_chromagram_act*2, 2000), self.nb_feature_chromagram))
        self.chromagram_est.fill(np.nan)
#         self.cum_distance = np.zeros((self.chromagram_est.shape[0], self.chromagram_est.shape[0]), dtype='float16')
        self.cum_distance = np.zeros((self.chromagram_act.shape[0], self.chromagram_est.shape[0]), dtype='float16')
        self.cum_distance.fill(np.nan)
                
        # Initialise the best_paths, in order to keep the best path at each iteration
        # (one iteration correspond to one update of the live feed)
        self.best_paths = []
        self.best_paths_distance = []
        
        # Check if we need to store the best paths
        self.use_low_memory = use_low_memory       
        
    def set_chromagram(self, wd, filename, chromagram_act=None):
        '''
        The function can take as input a 'wav' or a '.mid' file.
        in the latter case, we have to generate the corresponding '.wav'.  
                
        Get the chromagram from the disk or process the '.wav' file and write 
        the chromagram to disk if need be.
        '''
        
        # Build a specific key for the chromagram in case we want to store.
        suffix_filename = "_chromagram_S{}_H{}_fcn{}_mode{}_.npy".format(self.sr_act, 
                                                                         self.hop_length_act,
                                                                         self.compute_chromagram_fcn.__name__, 
                                                                         self.chromagram_mode)                                                                                                                                                                                                             
        
        filename_chomagram = wd + utils.unmake_file_format(filename, self.file_extension) + suffix_filename
        
        # We first check if a chromagram has been passed as input,
        # second, we check if the chromagram exists on disk, otherwise we generate it.
        if chromagram_act is not None:
            self.chromagram_act = chromagram_act         
        elif os.path.isfile(filename_chomagram) and self.store_chromagram:
            self.chromagram_act = np.load(filename_chomagram)
        else:
            # If the input file is a '.mid', we have to generate the '.wav'
            # Also, store the midi object, it will be useful to get the position in ticks
            if self.file_extension == '.mid':
                self.midi_obj = pretty_midi.PrettyMIDI(wd + filename)#filename_mid = filename                
                filename = utils.change_file_format(filename, '.mid', '.wav')                
                utils.write_wav(wd + filename, 
                                self.midi_obj.fluidsynth(self.sr_act, start_new_process32=utils.fluidsynth_start_new_process32()), 
                                rate = self.sr_act)
            
            # Now we are sure to have a '.wav', we can retrieve it.
            audio_data_wav = lb.core.load(wd + utils.make_file_format(filename, '.wav'), sr = self.sr_act)
            
            # Turn into chromagram
            self.chromagram_act = self.compute_chromagram(audio_data_wav[0])
            
            # Store if need be.                                                                                                                                                                                         
            if self.store_chromagram:                                                 
                np.save(filename_chomagram, self.chromagram_act)            
        

        self.len_chromagram_act_sec = self.chromagram_act.shape[0] * self.hop_length_act / float(self.sr_act)
        self.len_chromagram_act = self.chromagram_act.shape[0]
        self.nb_feature_chromagram = self.chromagram_act.shape[1]
        
    def compute_chromagram(self, y):
        '''
        Wrapper function to compute the chromagram.
        mode = 0: Return the raw chromagram
        mode = 1: Return the d_chromagram, i.e. the positive (time) difference
        mode = 2: Return [chromagram, d_chromagram] 
        '''
        
        if not (self.chromagram_mode==0 or self.chromagram_mode==1 or self.chromagram_mode==2): 
            raise ValueError('Mode needs to be in {0, 1, 2}')

        chromagram = self.compute_chromagram_fcn(y=np.array(y),
                                                 sr=self.sr_act,
                                                 hop_length=self.hop_length_act,
                                                 tuning=0.0,
                                                 **self.compute_chromagram_fcn_kwargs).T
                                            
        if self.chromagram_mode == 0:
            return(chromagram)
        else:
            d_chromagram = np.diff(chromagram, axis=0)
            d_chromagram = np.maximum(d_chromagram, 0.0)
            if self.chromagram_mode == 1:
                return(d_chromagram)
            else:
                return(np.hstack((chromagram[1:,:], d_chromagram)))        
        
    def select_advance_direction(self):        
        
        idx_est = self.idx_est
        idx_act = self.idx_act
            
        if idx_est < self.min_data or idx_act < self.min_data:
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
            return(CellList([i],[j-1]))            
            
        if j-1 < 0:
            return(CellList([i-1],[j]))
        
        isnan_s = np.isnan(self.cum_distance[i-1, j])
        isnan_w = np.isnan(self.cum_distance[i, j-1])
        isnan_sw = np.isnan(self.cum_distance[i-1, j-1])       
        
        # Standard case: compute everything        
        if not isnan_s and not isnan_w and not isnan_sw:
            return(CellList([i-1, i, i-1],[j, j-1, j-1]))
        
        # Edge case: nothing to compute.  
        elif isnan_s and isnan_w and isnan_sw:
            print('Could not find any valid cell leading to  {}, {}'.format(i, j))
            return(CellList([], []))
        
        # Otherwise, find the relevant cells to compute
        elif isnan_s and isnan_sw:
            return(CellList([i],[j-1]))
        
        elif isnan_w and isnan_sw:
            return(CellList([i-1],[j]))
        
        elif isnan_s:
            return(CellList([i,i-1], [j-1, j-1]))
        
        elif isnan_w:
            return(CellList([i-1, i-1],[j, j-1]))

                                                                
    def find_best_path(self, cells, return_distance):
        '''
        The function computes the local DTW path, adjusting for the weigths.
        It relies on the cells being passed such that the diagonal is the 
        last item (if two or three elements).
        
        It can either return the min distance or the min path.
        '''
                    
        if cells.len == 3:
            d = np.array([self.cum_distance[cells.get_cell_as_tuple(0)], 
                          self.cum_distance[cells.get_cell_as_tuple(1)], 
                          self.diag_cost * self.cum_distance[cells.get_cell_as_tuple(2)]])            
                    
        elif cells.len == 2:            
            d = np.array([self.cum_distance[cells.get_cell_as_tuple(0)], self.diag_cost * self.cum_distance[cells.get_cell_as_tuple(1)]])
            
        elif cells.len == 1:
            d = np.array([self.cum_distance[cells.get_cell_as_tuple(0)]])
            
        else:
            d = np.array([np.nan])
        
        argmin = np.argmin(d)
        
        if return_distance:                
            return(d[argmin])
        else:
            return(cells.get_cell(argmin))
                            
    def update_cum_distance(self, direction):
        '''
        Taking the existing cum_distance matrix and the search direction, this function
        updates the cum_distance by finding the (local) min distance path.
        '''
        
        # For the first iteration, just take the distance between the two chromagrams
        if self.idx_act == -1 and self.idx_est == -1:
            self.idx_act += 1
            self.idx_est += 1
            self.cum_distance[self.idx_act, self.idx_est] = utils.distance_chroma(self.chromagram_act[self.idx_act, :], 
                                                                                  self.chromagram_est[self.idx_est, :])
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
            
            for k in np.arange(max(idx_est-self.search_width+1, 0), idx_est+1):                
                distance = utils.distance_chroma(self.chromagram_act[idx_act, :], self.chromagram_est[k, :])                
                cells = self.find_cells(idx_act, k)
                self.cum_distance[idx_act, k] = distance + self.find_best_path(cells, True)                                                                 
            return
            
        if direction == 2:
            # Advance the live feed
            self.idx_est += 1
            self.input_advanced = True  
            idx_act = self.idx_act
            idx_est = self.idx_est

#             RESTORE
#             for k in np.arange(max(idx_act-self.search_width+1, 0), idx_act+1):
#                 distance = utils.distance_chroma(self.chromagram_act[k, :], self.chromagram_est[idx_est, :])
#                 cells = self.find_cells(k, idx_est)
#                 self.cum_distance[k, idx_est] = distance + self.find_best_path(cells, True)
#             return
            min_idx_act = max(idx_act-int(ceil(self.search_width/2.0))+1, 0)
            max_idx_act = min(idx_act+int(ceil(self.search_width/2.0))+1, self.len_chromagram_act)
            for k in np.arange(min_idx_act, max_idx_act):                
                distance = utils.distance_chroma(self.chromagram_act[k, :], self.chromagram_est[idx_est, :])
                cells = self.find_cells(k, idx_est)
                self.cum_distance[k, idx_est] = distance + self.find_best_path(cells, True)
            return


    def update_best_path(self):
        '''
        Based on the cum distance matrix, this function finds the best path
        Running this function is not required for the main loop, it serves mainly 
        for ex-post analysis.
        ''' 

        idx_est = self.idx_est               
        idx_act = np.nanargmin(self.cum_distance[0:self.idx_act+1,idx_est])

        best_path = CellList([idx_act], [idx_est])
        best_path_distance = []       
            
        # Iterate until the starting point
        while idx_act > 0 or idx_est > 0:
            best_path_distance.append(self.cum_distance[idx_act,idx_est])
            cells = self.find_cells(idx_act, idx_est)  
            best_path_local = self.find_best_path(cells, False)
            (idx_act, idx_est) = best_path_local.get_cell_as_tuple(0) 
            best_path.append(best_path_local)
            
        # Keep the best path for successive iterations
        # (the history of the best paths could be dropped at a later stage) 
        self.best_paths.append(best_path)
        self.best_paths_distance.append(list(reversed(best_path_distance)))      
        
    def update_position(self):        
        '''
        Append idx_act to the list of estimated positions (and the equivalent in seconds). 
        In the case we have the midi_object available, also report the position in ticks
        '''
        
        self.position.append(self.idx_act) # RESTORE, MAYBE.... 
#         self.position.append(np.nanargmin(self.cum_distance[0:self.idx_act+1,self.idx_est]))
#         self.position.append(np.nanargmin(self.cum_distance[:,self.idx_est])) # RESTORE!!!!!.... 
        self.position_sec.append(self.position[-1] * self.hop_length_act / float(self.sr_act))
        
        # Only possible if the input file was a '.mid' in the first place
        if self.file_extension == '.mid':
            self.position_tick = self.midi_obj.time_to_tick(self.position_sec[-1])
        
    def plot_dtw_distance(self, paths=[-1]):
        '''
        Plot the cumulative distance matrix as a heat map and the best path.
        
        path_nb is the number of the path we want to plot (-1 for the final path). 
        '''
        utils.plot_dtw_distance(self.cum_distance)

        for k in paths:
            plt.plot(self.best_paths[k].cols, self.best_paths[k].rows, color='black')
            
    def plot_chromagrams(self):
        """
        Plot both the estimated and the actual chromagram for comparison.
        """
        
        axarr = plt.subplots(2, 1)[1]
        
        # Plot chromagram_act
        utils.plot_chromagram(self.chromagram_act, sr=self.sr_act, hop_length=self.hop_length_act, ax=axarr[0], xticks_sec=True)
        axarr[0].set_title('Chromagram act')
        
        # Plot chromagram_est (up to idx_est). Will need to amend to sr_est and hop_length_est at some point...
        utils.plot_chromagram(self.chromagram_est[0:self.idx_est,:], sr=self.sr_act, hop_length=self.hop_length_act, ax=axarr[1], xticks_sec=True)
        axarr[1].set_title('Chromagram est')   
        
    def match_batch(self, audio_data_est):
        '''
        Align a target raw audio.
        The raw audio needs to be sampled as per the matcher's sample rate.
        Though the process is offline because we pass in the entire audio set, the 
        actual alignment does not use any forward-looking data.
        
        Return a mapping between the times of the target audio vs times of the "true" audio.  
        '''
        
        # Compute the chromagram, use the engine of the matcher to ensure that both chromagrams have 
        # been computed with the same procedure.            
        chromagram_est = self.compute_chromagram(audio_data_est)                                                                    
         
        # Run the online alignment, frame by frame
        nb_frames_est = chromagram_est.shape[0]
        for n in range(nb_frames_est):
            self.main_matcher(chromagram_est[n,:])
            
        # Get the alignment output
        # Not sure whether we should add/take out one frame... Also, we should use matcher_tmp.sr_est once possible
        times_cor_est = np.arange(nb_frames_est) * self.hop_length_act / float(self.sr_act) 
        times_ori_est = np.array(self.position_sec)
        
        return(times_cor_est, times_ori_est)
    
    def match_offline(self, audio_data_est):
        '''
        Use the librosa DTW implementation to perform offline matching.
        '''
        
        # Compute the chromagram, use the engine of the matcher to ensure that both chromagrams have 
        # been computed with the same procedure.            
        chromagram_est = self.compute_chromagram(audio_data_est)
        
        # Set up the diagonal cost        
        weights_mul = np.array([1.0*self.diag_cost, 1.0, 1.0])   
        
        # Run the DTW alignment
        [cum_distance, best_path] = lb.core.dtw(self.chromagram_act.T, chromagram_est.T, weights_mul=weights_mul)
        
        # Get the alignment output
        times_ori_est = best_path[:,0] * self.hop_length_act / float(self.sr_act)         
        times_cor_est = best_path[:,1] * self.hop_length_act / float(self.sr_act)
                            
        return(times_cor_est, times_ori_est, {'cum_distance':cum_distance, 'best_path':best_path})                                
                                                        
    def main_matcher(self, chromagram_est_row): 
        '''
        Called for every new frame of the live feed, for which we run the online DTW
        '''
        
        # Store the new chromagram
        self.chromagram_est[self.idx_est+1, :] = chromagram_est_row
                                    
        self.input_advanced = False
        # Run the main loop
        while not self.input_advanced:
            
            # Disable the matching procedure if we are at the end of the act data
            # In that case, we keep updating the best path (keeping idx_act to the last act value)
            if self.idx_act >= self.len_chromagram_act - 1:
                self.idx_est += 1 # Increment the estimated position nonetheless
                self.update_position()
                return
                     
            direction = self.select_advance_direction()
            self.update_cum_distance(direction)        
            
        # Find the best path and the current position
        # Do not run if we are at the starting point
        if (self.idx_act > 0 or self.idx_est > 0):                    
            
            # Update the estimated current position
            self.update_position()  
            
            if not self.use_low_memory:                
                self.update_best_path()            