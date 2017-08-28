import librosa as lb
import numpy as np
import corrupt_midi
import os
import utils_audio_transcript as utils
import pretty_midi
from midi.utils import midiread, midiwrite
from math import floor, ceil
import dtw
import copy
import matplotlib.pyplot as plt
import config
            
class MatcherMidi():
    '''
    Offline DTW to align MIDI files. The mid-level feature used is the piano-roll representation
    (does not require the used instrument to be a piano)    
    '''
     
    def __init__(self, wd, filename1, filename2):
         
        self.midi_band = (0, 127) # the range of possible Midi notes
        self.dt = 0.1 # the time resolution
         
        # Set up the mid-level features used for matching 
        self.piano_roll1 = midiread(wd + filename1, r=self.midi_band, dt=self.dt).piano_roll
        self.piano_roll2 = midiread(wd + filename2, r=self.midi_band, dt=self.dt).piano_roll
         
    def build_aligned_midi(self, wd, filename1, filename2):
         
        # Reconstruct the aligned piano rolls   
        self.aligned_piano_roll1 = self.piano_roll1[self.path[0],:]
        self.aligned_piano_roll2 = self.piano_roll2[self.path[1],:]
         
        # Reconstruct the aligned midis      
        midiwrite(wd + filename1, self.aligned_piano_roll1, r=self.midi_band, dt=self.dt)
        midiwrite(wd + filename2, self.aligned_piano_roll2, r=self.midi_band, dt=self.dt)
         
         
    def main_matcher(self):
                 
        # Run the DTW matching algorithm
        distance_tot, distance_matrix, best_path = dtw.dtw(self.piano_roll1, self.piano_roll2, utils.distance_midi_cosine)
        self.distance_tot = distance_tot
        self.distance_matrix = distance_matrix
        self.path = best_path       
        
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
    
#     def __init__(self, wd, filename, sr, n_fft, hop_length, 
#                  diag_cost=1.2,
#                  compute_chromagram_fcn = lb.feature.chroma_stft,
#                  compute_chromagram_fcn_kwargs = {}, 
#                  use_low_memory=True):
    def __init__(self, wd, filename, sr, hop_length, 
                 diag_cost=1.2,
                 compute_chromagram_fcn = lb.feature.chroma_stft,                 
                 compute_chromagram_fcn_kwargs = {}, 
                 chromagram_mode = 0,
                 use_low_memory=True): 
        
        # Start by fixing the sample rate and hop size for the spectrum
        # decomposition
        self.sr_act = sr
#         self.n_fft_act = n_fft
        self.hop_length_act = hop_length   
        
        # Set the function used to compute the chromagram and its parameters
        self.compute_chromagram_fcn = compute_chromagram_fcn
        self.compute_chromagram_fcn_kwargs = compute_chromagram_fcn_kwargs
        self.chromagram_mode = chromagram_mode
        
        # Check if we need to store the chromagram to disk and retrieve it when it exists
        self.store_chromagram = False  
        
        # Load the .wav file and turn into chromagram
        self.set_chromagram(wd, filename)
                
        # Current position in the act / est data
        # Set to -1 as no data has been processed
        self.idx_act = -1
        self.idx_est = -1
        
        # position and position_sec represent the expected position in the Midi file
        self.position = [0]
        self.position_sec = [0]
        
        # DTW parameters
        self.search_width = 100 # width of the band to search the optimal alignment (in frames)
        self.min_data = 20 # min number of input frames to compute before doing anything        
        self.diag_cost = diag_cost # the cost for the diagonal (1.0<=x<=2.0, may be set to less than 2 to let the algorithm favour diagonal moves) 
        
        # Boolean to check if the new input has been processed
        self.input_advanced = False
        
        # Initialise large empty matrices for chromagram_est and for the cumulative distance matrix
        self.chromagram_est = np.zeros((self.len_chromagram_act*2, self.nb_feature_chromagram))
        self.chromagram_est.fill(np.nan)
        self.cum_distance = np.zeros((self.chromagram_est.shape[0], self.chromagram_est.shape[0]), dtype='float16')
        self.cum_distance.fill(np.nan)
        
        # Initialise the best_paths, in order to keep the best path at each iteration
        # (one iteration correspond to one update of the live feed)
        self.best_paths = []
        
        # Check if we need to store the best paths
        self.use_low_memory = use_low_memory
        

        
    def set_chromagram(self, wd, filename):
        '''        
        Get the chromagram from the disk or process the .wav file and write 
        the chromagram to disk if need be.
        '''        

        suffix_filename = "_chromagram_S{}_H{}_fcn{}_mode{}_.npy".format(self.sr_act, 
                                                                         self.hop_length_act,
                                                                         self.compute_chromagram_fcn.__name__, 
                                                                         self.chromagram_mode)
                                                                                                                                                                                                             
        
        filename_chomagram = wd + utils.unmake_file_format(filename, '.wav') + suffix_filename
        
        if os.path.isfile(filename_chomagram) and self.store_chromagram:
            self.chromagram_act = np.load(filename_chomagram)
        else:
            audio_data_wav = lb.core.load(wd + utils.make_file_format(filename, '.wav'), sr = self.sr_act)
#             self.chromagram_act = Matcher.compute_chromagram(audio_data_wav[0], 
#                                                              self.compute_chromagram_fcn, self.compute_chromagram_fcn_kwargs,                                                             
#                                                              self.sr_act, self.n_fft_act, self.hop_length_act)
            self.chromagram_act = Matcher.compute_chromagram(audio_data_wav[0], 
                                                             self.sr_act,
                                                             self.hop_length_act,
                                                             self.compute_chromagram_fcn, 
                                                             self.compute_chromagram_fcn_kwargs,
                                                             self.chromagram_mode)                                                                                                                                                                                         
            if self.store_chromagram:                                                 
                np.save(filename_chomagram, self.chromagram_act)            
        

        self.len_chromagram_act_sec = self.chromagram_act.shape[0] * self.hop_length_act / float(self.sr_act)
        self.len_chromagram_act = self.chromagram_act.shape[0]
        self.nb_feature_chromagram = self.chromagram_act.shape[1]
        
    @staticmethod
    def compute_chromagram(y, sr, hop_length, compute_chromagram_fcn, compute_chromagram_fcn_kwargs, mode):
        '''
        Wrapper function to compute the chromagram.
        mode = 0: Return the raw chromagram
        mode = 1: Return the d_chromagram, i.e. the positive (time) difference
        mode = 2: Return [chromagram, d_chromagram] 
        '''
        
        if not (mode==0 or mode==1 or mode==2): 
            raise ValueError('Mode needs to be in {0, 1, 2}')

        chromagram = compute_chromagram_fcn(y=np.array(y),
                                            sr=sr,
                                            hop_length=hop_length,
                                            tuning=0.0,
                                            **compute_chromagram_fcn_kwargs).T
                                            
        if mode == 0:
            return(chromagram)
        else:
            d_chromagram = np.diff(chromagram, axis=0)
            d_chromagram = np.maximum(d_chromagram, 0.0)
            if mode == 1:
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
        
#         # The case with all 3 nans should not arise
#         if isnan_s and isnan_w and isnan_sw: 
#             raise ValueError('Could not find any valid cell leading to ({}, {})'.format(i, j))        
        
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
            
            for k in np.arange(max(idx_act-self.search_width+1, 0), idx_act+1):
#             for k in np.arange(max(idx_act-self.search_width/2+1, 0), min(idx_act+self.search_width/2+1, self.len_chromagram_act)): # VERIFY!!!
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
            
        # Iterate until the starting point
        while idx_act > 0 or idx_est > 0:
            cells = self.find_cells(idx_act, idx_est)  
            best_path_local = self.find_best_path(cells, False)
            (idx_act, idx_est) = best_path_local.get_cell_as_tuple(0) 
            best_path.append(best_path_local)
            
        # Keep the best path for successive iterations
        # (the history of the best paths could be dropped at a later stage) 
        self.best_paths.append(best_path)        
        
    def update_position(self):        
        '''
        Append idx_act to the list of estimated positions (and the equivalent in seconds). 
        '''
        
        self.position.append(self.idx_act)
        self.position_sec.append(self.position[-1] * self.hop_length_act / float(self.sr_act))
        
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
            

class MatcherEvaluator():
    """
    Class to evaluate the score following procedure.
    1) Apply several corruptions to the original Midi file
    2) Perform alignment for each
    3) Compute the difference between ground-truth and estimated positions
    """     
    def __init__(self, wd, filename_midi, config_matcher={}, configs_corrupt={}):
        
        self.wd = wd
        self.filename_midi_act = filename_midi
        self.sr = 11025 # Used for both synthesis of the Midi files and alignment
        self.midi_object = pretty_midi.PrettyMIDI(self.wd + self.filename_midi_act) # Keep the original Midi file in memory 
        self.len_act_sec = self.midi_object._PrettyMIDI__tick_to_time[-1]
        self.times_ori_act = np.linspace(0.0, self.len_act_sec, int(self.len_act_sec/0.01)) # The reference timestamps in the original data
        self.times_est_all = [] # Placeholder for the output times of the alignment procedure [corrupted, original]
        self.times_cor_act_all = [] # Placeholder for the corrupted times (ground-truth)
        self.alignement_stats = [] # Placeholder for the output of the evaluation procedure
                 
        # Retrieve the corruption configs
        self.corrupt_configs = configs_corrupt

        self.filenames_wav_corrupt = [] # Placeholder the list of corrupted filenames
        self.matchers = [] # Placeholder for the matchers. May be removed at a later stage (for reporting only)...
        self.times_rests = [] # Placeholder for the starting and ending rests of the corrupted track
        self.config_matcher = config_matcher # Config to override the matcher's parameters  
        
    def corrupt(self):
        '''
        Loop over the corruption configs, apply these to the Midi, synthetise and store as .wav.
        '''
        
        for cnt, config in enumerate(self.corrupt_configs):
            # Make a copy of the midi object as corrupt_midi.corrupt_midi will modify it
            midi_object_corrupt = copy.deepcopy(self.midi_object)
            
            # Apply the corruption procedure
            times_cor_act, diagnostics = corrupt_midi.corrupt_midi(midi_object_corrupt, self.times_ori_act, **config)
            self.times_cor_act_all.append(times_cor_act)
            
            # Synthetise the .wav file
            audio_data = midi_object_corrupt.fluidsynth(self.sr, start_new_process32=True)
            
            # Store the .wav data
            filename_wav_corrupt = utils.change_file_format(self.filename_midi_act, '.mid', '.wav', append = '_{}'.format(cnt))
            self.filenames_wav_corrupt.append(filename_wav_corrupt) # Keep the names for later use            
            utils.write_wav(self.wd + filename_wav_corrupt, audio_data, rate = self.sr)
            
            # Store the .mid (not required)
            midi_object_corrupt.write(self.wd + self.filename_midi_act[:len(self.filename_midi_act)-4] + '_{}.mid'.format(cnt))            
            
    def align(self):
        '''
        Perform the alignement procedure between the original .wav and each of the corrupted .wav.
        Keep the results of the alignment procedure.
        '''
        
#         n_fft = 2048
        hop_length = 1024
        
        # First, turn the original .mid file into .wav
        self.filename_wav_act = utils.change_file_format(self.filename_midi_act, '.mid', '.wav')
        utils.write_wav(self.wd + self.filename_wav_act, pretty_midi.PrettyMIDI(self.wd + self.filename_midi_act).fluidsynth(self.sr, start_new_process32=True), rate = self.sr)
        
        # Initialise the matcher
#         matcher = Matcher(self.wd, self.filename_wav_act, self.sr, n_fft, hop_length, **self.config_matcher)
        matcher = Matcher(self.wd, self.filename_wav_act, self.sr, hop_length, **self.config_matcher)
        
        # Loop over the configs
        for filename_wav_corrupt in self.filenames_wav_corrupt:
            
            print('Aligning {}'.format(filename_wav_corrupt))
            
            # Copy the matcher as it will be modified
            matcher_tmp = copy.deepcopy(matcher)
            
            # Load the audio data
            audio_data_est = lb.core.load(self.wd + filename_wav_corrupt, sr = matcher_tmp.sr_act)[0]
            
            # Find the starting and ending rests
            rests = utils.find_start_end_rests(audio_data_est, matcher_tmp.sr_act) # No need to pass in hop_size and n_fft
            self.times_rests.append(rests)
            
            # Compute the chromagram, use the engine of the matcher to ensure that both chromagrams have 
            # been computed with the same procedure.
#             chromagram_est = Matcher.compute_chromagram(audio_data_est, 
#                                                         matcher_tmp.compute_chromagram_fcn, 
#                                                         matcher_tmp.compute_chromagram_fcn_kwargs, 
#                                                         self.sr, n_fft, hop_length)

            chromagram_est = Matcher.compute_chromagram(audio_data_est, 
                                                        self.sr, 
                                                        hop_length,                                                        
                                                        matcher_tmp.compute_chromagram_fcn, 
                                                        matcher_tmp.compute_chromagram_fcn_kwargs,
                                                        matcher_tmp.chromagram_mode)                                                         
            
            # Run the online alignment, frame by frame
            nb_frames_est = chromagram_est.shape[0]
            for n in range(nb_frames_est):
                matcher_tmp.main_matcher(chromagram_est[n,:])
                
            # Keep the alignment output
            # Not sure whether we should add/take out one frame... Also, we should use matcher_tmp.sr_est once possible
            times_cor_est = np.arange(nb_frames_est) * matcher_tmp.hop_length_act / float(matcher_tmp.sr_act) 
            times_ori_est = np.array(matcher_tmp.position_sec)
            self.times_est_all.append(np.array([times_cor_est, times_ori_est]).T)
            
            # Keep the matchers (for reporting only, may be removed at a later stage)
            self.matchers.append(matcher_tmp)            
                
    def evaluate(self):
        '''
        Loop over the corruption configs and calculate the alignment stats.
        '''
        for cnt, times_est in enumerate(self.times_est_all):
            
            times_cor_est = times_est[:,0]
            times_ori_est = times_est[:,1]            
            
            # Compute the alignment error for all the times_cor_est times.
            alignment_error_raw = utils.calc_alignment_stats(times_cor_est, times_ori_est, self.times_cor_act_all[cnt], self.times_ori_act)
            
            # Extract the values that correspond to the times after the starting rest and before 
            # the finishing rest.  
            mask = np.logical_and(times_cor_est >= self.times_rests[cnt][0], times_cor_est <= self.times_rests[cnt][1])               
            alignment_error = alignment_error_raw[mask]
            
            # Compute the aggregate metrics
            mean_error = np.mean(alignment_error)
            mean_abs_error = np.mean(np.absolute(alignment_error))
            prctile_error = np.percentile(alignment_error, np.array([0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0]))
                                       
            alignement_stats = {
                'alignment_error_raw':alignment_error_raw,
                'alignment_error':alignment_error,
                'mean_error':mean_error,
                'mean_abs_error':mean_abs_error,
                'prctile_error':prctile_error,
                'idx_config_corruption':cnt                              
                } 
             
            self.alignement_stats.append(alignement_stats)
            
    def plot_alignment_error(self):
        '''
        Plot the alignment error for all the corruption configs.
        '''
        nbConf = len(self.corrupt_configs)
        axarr = plt.subplots(int(ceil(nbConf/2.0)), min(nbConf, 2), squeeze=False)[1]
        for cnt, config in enumerate(self.corrupt_configs):          
            ax = axarr[int(floor(cnt/2.0)), cnt%2 ]                
            ax.plot(self.alignement_stats[cnt]['alignment_error_raw'])
            ax.set_title(str(config))
        
    def main_evaluation(self):
        self.corrupt()
        self.align()
        self.evaluate()
