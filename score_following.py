import librosa as lb
import numpy as np
import os
import utils_audio_transcript as utils
import pretty_midi
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.spatial.distance import cdist
from dtw_fast import dtw_fast
            
class OnlineLocalDTW():
    """
    Online Local DTW. This is the top-level function, it only does 
    the management of the input data. We call the cython implementation of
    the DTW algorithm. We only run alignment on a sub-part of the cost matrix.    
    We allow for subsequence alignment. There is no continuity constraint.    
    """
    def __init__(self,
                 features_act,
                 nb_frames_dtw_est,  
                 diag_cost=1.0):                
        
        # Input features, the ones we want to track
        self.features_act = features_act 
        self.nb_frames_feature_act = features_act.shape[0]
        self.nb_bin_feature = features_act.shape[1]        
        
        # Initialise position and position_sec and position_tick.  
        # They represent the expected position in the Midi file.
        # The best distance is the total alignment distance (cum or EWMA)
        # We have:  best_distance = [x[-1] for x in best_paths_distance] 
        self.position = 0
        self.positions = []
        
        # Initialise estimated idx (idx_est represent how many frames of the 
        # live feed we have processed).
        self.idx_est = -1

        # DTW parameters
        self.nb_frames_dtw_est = nb_frames_dtw_est 
        self.diag_cost = diag_cost
        self.weights_mul = np.array([1.0*diag_cost, 1.0, 1.0])                  
        
        # Initialise large empty matrices for features_est and for the cumulative distance matrix
        self.features_est = np.zeros((max(self.nb_frames_feature_act*2, 2000), self.nb_bin_feature))
        self.features_est.fill(np.nan)
        self.distance = np.zeros((self.nb_frames_feature_act, self.features_est.shape[0]), dtype='float32')
        self.distance.fill(np.nan)
        self.cum_distance = np.zeros((self.nb_frames_feature_act, self.features_est.shape[0]), dtype='float32')
        self.cum_distance.fill(np.nan)                                        
        
    def main_dtw(self, features_est_new, idx_act_lb, idx_act_ub, max_consecutive_steps):
        '''
        Called for every new frame of the live feed, for which we run the online local DTW.
        '''
        
        # Perform some basic checks on the input
        if features_est_new.shape != (self.nb_bin_feature,):
            raise(ValueError('Incompatible shape, expect 1D array with {} items'.format(self.nb_bin_feature)))
        
        if (idx_act_lb < 0 or idx_act_ub < 0 or idx_act_lb > idx_act_ub or             
            idx_act_lb >= self.nb_frames_feature_act or idx_act_ub >= self.nb_frames_feature_act):           
            raise(ValueError(b'Incompatible bounds ({}/{})'.format(idx_act_lb, idx_act_ub)))        
        
        # Increment the live feed index
        self.idx_est += 1        
        
        # Store the new features
        self.features_est[self.idx_est, :] = features_est_new
        
        # Update the distance matrix 
        # We use the L2 norm to compute the local distance (frame / frame) 
        self.distance[:, self.idx_est] = np.reshape(cdist(self.features_act, np.atleast_2d(features_est_new)),-1,1)
        
        # Extract the distance block that we care about
        distance_tmp = self.distance[idx_act_lb:idx_act_ub+1, max(self.idx_est-self.nb_frames_dtw_est+1, 0):self.idx_est+1]
        
        # Run the DTW with subsequence alignment (cython)
        cum_distance_tmp = dtw_fast(np.array(distance_tmp, dtype=np.float64) , self.weights_mul, 1, max_consecutive_steps)
        
        # Keep the last column of the alignment distance only 
        cum_distance_last = cum_distance_tmp[:,-1]
        cum_distance_last[np.isinf(cum_distance_last)] = np.NaN
        self.cum_distance[idx_act_lb:idx_act_ub+1, self.idx_est] = cum_distance_tmp[:,-1]

class MatcherFilter():
    '''
    Filter the output of the DTW procedure.
    Also compute the search bounds for the DTW.
    '''
    
    def __init__(self, nb_frames_act, nb_frames_dtw_est, nb_frames_dtw_act):
                    
        # Number of obs in the act features
        self.nb_frames_act = nb_frames_act
        
        # Speed is the nb of act frames processed for one est frame
        self.speed = 1.0         
        self.min_speed = 1.0/3.0
        self.max_speed = 3.0      
        self.nb_frames_speed_estimate = 50        
        self.positions_filtered = []
        self.min_frames_position_base = 50
        self.positions_base = []
        self.idx_est = -1
        self.nb_frames_dtw_est = nb_frames_dtw_est # nb frames of est data dtw uses 
        self.nb_frames_dtw_act = nb_frames_dtw_act # nb frames of act data dtw uses
        self.idx_act_lb = 0
        self.idx_act_ub = self.idx_act_lb + int(ceil(self.nb_frames_dtw_act/2.0))
        
        # Smoothing parameters for the base position. 
        half_life_speed_estimate = 10
        half_life_position_base = 10
        self.alpha_speed_estimate = pow(0.5, 1.0/half_life_speed_estimate) 
        self.alpha_position_base = pow(0.5, 1.0/half_life_position_base)
        
        # The local step constraints for the DTW, that is, the maximum number
        # of consecutive step that we can do in either the est of the act direction.
        self.max_consecutive_steps_init = 1000
        self.max_consecutive_steps_real = 3
        self.max_consecutive_steps = self.max_consecutive_steps_init    
        
        # We only want to apply the local constraint once the following
        # has properly started. In the case where, the act feed starts later 
        # than the est feed, we don't want the local constraint to force moving forward. 
        self.start_max_consecutive_steps = 50
        
        # Confidence calculation parameters
        self.nb_frames_confidence_estimate = [30, 50, 100]
        self.high_confidence_alignment = False               
        
    def update_speed(self):
        '''
        Estimate the speed at which the live feed is going (relative to the act data).
        We could also rely on tempo estimation techniques.
        Cap and floor the result.
        '''
        
        if self.idx_est < self.nb_frames_dtw_est + self.nb_frames_speed_estimate:                        
            return                  
        
        speed = np.mean(np.diff(self.positions_filtered_last)) # The mean may not be very robust
        self.speed = self.alpha_speed_estimate * self.speed + (1-self.alpha_speed_estimate) * speed
        
        self.speed = max(min(self.speed, self.max_speed), self.min_speed)  
        
    def update_position_base(self):
        '''
        Update the base position, which it the slow-moving / benchmark position.
        We use it to get the search bounds. 
        '''        
        if self.idx_est < self.min_frames_position_base:
            self.positions_base.append(self.positions_base[-1] + self.speed if self.idx_est>0 else 0)
        else:
            position_filtered_new = np.mean(self.positions_filtered_last) + self.speed*self.nb_frames_speed_estimate/2.0
            position_base_new = self.positions_base[-1] + self.speed             
            self.positions_base.append(self.alpha_position_base * position_base_new + (1-self.alpha_position_base) * position_filtered_new)
                    
    def update_search_bounds(self):
        '''
        The search bounds define the area over which we want to perform the 
        alignment in the act data.          
        '''
        
        self.idx_act_lb = max(int(round(self.positions_base[-1])) - int(floor(self.nb_frames_dtw_act/2.0)), 0)
        self.idx_act_ub = min(int(round(self.positions_base[-1])) + int(ceil(self.nb_frames_dtw_act/2.0)), self.nb_frames_act - 1)
        
    def update_max_consecutive_steps(self):
        '''
        We constrain the number of consecutive steps only if the alignment has
        properly started. 
        '''        
        if self.positions_base[-1] >= self.start_max_consecutive_steps:
            self.max_consecutive_steps = self.max_consecutive_steps_real
        else:  
            self.max_consecutive_steps = self.max_consecutive_steps_init
    
    def update_confidence(self):
        '''
        Estimate how much confidence we may put in the recent alignment.
        '''
        self.high_confidence_alignment = False
        
        for nb_frames in self.nb_frames_confidence_estimate:
            
            if self.idx_est >= nb_frames:                                
                confidence_x = np.array([np.arange(nb_frames), np.ones(nb_frames)]).T
                confidence_y = np.array(self.positions_filtered[len(self.positions_filtered)-nb_frames:])
                confidence_beta = np.linalg.lstsq(confidence_x, confidence_y)[0]
                confidence_residuals = confidence_y - np.dot(confidence_x, confidence_beta)
                confidence_bmk_residuals = confidence_y - np.mean(confidence_y) 
                confidence_rsq = 1 - np.sum(np.power(confidence_residuals, 2)) / np.sum(np.power(confidence_bmk_residuals, 2))
                
                if confidence_beta[0] > self.min_speed and confidence_beta[0] < self.max_speed and confidence_rsq > 0.7:
                    self.high_confidence_alignment = True
                       

    def filter_position(self, cum_distance):        
        '''
        Taking the DTW distance as input, run some heuristics to identify our best guess of 
        the current position. In essence, we look for the point with the minimum DTW distance. 
        '''
        cum_distance_tmp = cum_distance[:, self.idx_est]
        curr_position = self.positions_filtered[-1] if self.idx_est>0 else 0
        relative_mins_idxs, relative_mins_values = utils.find_relative_mins(np.atleast_2d(cum_distance_tmp).T, 10, 5)
         
        relative_mins_idxs = relative_mins_idxs[~np.isnan(relative_mins_idxs)].astype(np.int32)
        relative_mins_values = relative_mins_values[~np.isnan(relative_mins_values)]        
        
        nearest_position_tmp = np.nanargmin(np.abs(curr_position - relative_mins_idxs))
        nearest_position_idx = relative_mins_idxs[nearest_position_tmp]
        nearest_position_value = relative_mins_values[nearest_position_tmp]
        # TODO : COMMENT AND REMOVE FIXED VALUE
        if self.high_confidence_alignment and nearest_position_value < 37:            
            position_filtered_new = nearest_position_idx             
        else:
            position_filtered_new = relative_mins_idxs[np.nanargmin(relative_mins_values)]
         
        self.positions_filtered.append(position_filtered_new)         
        
    def main_filter(self, cum_distance):
        '''
        Top-level function for the filtering.
        Called for every new frame of est data.
        '''
        
        # Increment the estimated position
        self.idx_est += 1
        
        # Keep the last filtered positions
#         self.positions_filtered_last = np.array(self.positions_filtered[max(self.idx_est - self.nb_frames_speed_estimate + 1, 0):self.idx_est+1])
        self.positions_filtered_last = np.array(self.positions_filtered[max(self.idx_est - self.nb_frames_speed_estimate, 0):self.idx_est])
        
        self.update_speed()
        self.update_position_base()
        self.update_search_bounds()
        self.update_confidence()
        self.filter_position(cum_distance)
        self.update_max_consecutive_steps()        
                                                                                                
class Matcher():
    '''
    Top-level class for the score following procedure. 
    This class serves as a wrapper around the online DTW.
    '''
    def __init__(self, wd, filename, sr, hop_length,
                 compute_chromagram_fcn=lb.feature.chroma_stft,                
                 compute_chromagram_fcn_kwargs={}, 
                 chromagram_mode=0,
                 chromagram_act=None):        
        
        # Clean the filename
        filename_clean = os.path.normpath(filename)
        
        # Fix the sample rate and hop size for the spectrum decomposition
        self.sr = sr
        self.hop_length = hop_length  
        
        # Initialise the placeholder for the estimated audio data (only the most recent)
        # This is required in the online mode, as we may want to compute the chromagram  
        # with more data that just the last chunk.
        self.audio_data = np.array([])                        
        self.min_len_sample = utils.calc_nb_sample_stft(self.sr, self.hop_length, 3.0) # Nb of samples to compute the chromagram
        
        # Initialise another placeholder for audio data. 
        # Keep all the audio that has been input for alignment (used for offline analysis)
        self.audio_data_all = np.array([], dtype=np.float32) 
        
        # Set the function used to compute the chromagram and its parameters
        self.compute_chromagram_fcn = compute_chromagram_fcn
        self.compute_chromagram_fcn_kwargs = compute_chromagram_fcn_kwargs
        self.chromagram_mode = chromagram_mode
        
        # Check if we need to store the chromagram to disk and retrieve it when it exists
        self.store_chromagram = False  
        
        # Check whether we have the '.mid' or '.wav' as input.
        _, self.file_extension = os.path.splitext(filename_clean)        
        if not (self.file_extension == '.mid' or self.file_extension == '.wav'):
            raise ValueError('Input file need to be either .mid or .wav. Filename is {} while extension is {}'.format(filename_clean, self.file_extension))
        
        # Load the .wav file and turn into chromagram.
        if chromagram_act is None:
            chromagram_act = self.get_chromagram(wd, filename_clean)
            
        # Keep the number of frames of the act features
        self.nb_frames_feature_act = chromagram_act.shape[0]   

        # Place-holders for the position
        self.position_tick = 0
        self.positions_sec = []
        self.positions = []
        
        # Initialise the DTW engine
        nb_frames_dtw_est = 50
        nb_frames_dtw_act = 150
        max_consecutive_steps = 3
        dtw_fcn = dtw_fast
        distance_local_fcn = cdist
        self.dtw = OnlineLocalDTW(chromagram_act, nb_frames_dtw_est)
        
        # Initialise the filter
        self.filter = MatcherFilter(self.nb_frames_feature_act, nb_frames_dtw_est, nb_frames_dtw_act)
        
        # TODO : PUT IN PROPER FUNCTION, COMMENT AND STORE THE CHROMAGRAM
        samples_idxs = [4,12,21,28,36,37,45,48,51,52,53,54,59,61,69,75,76,87,89,91]
        wd_samples = utils.WD_AUDIOSET + r"VerifiedDataset//VerifiedDatasetRecorded//"
        nb_frames_per_sample = 100
        chromagram_sample = np.empty([nb_frames_per_sample*len(samples_idxs), chromagram_act.shape[1]])
        
        for k, idx in enumerate(samples_idxs):            
            audio_data_sample = lb.core.load("{}sample_{}.wav".format(wd_samples, idx), sr = self.sr)[0]
            chromagram_sample_tmp = self.compute_chromagram(audio_data_sample)  
            chromagram_sample[k*nb_frames_per_sample:(k+1)*nb_frames_per_sample, :] = chromagram_sample_tmp[0:nb_frames_per_sample, :] 
                
        nb_boot = 500
        boot_idx_act = np.random.randint(0, self.nb_frames_feature_act - nb_frames_dtw_act - 1, nb_boot)
        boot_idx_est = np.random.randint(0, chromagram_sample.shape[0] - nb_frames_dtw_act - 1, nb_boot)
        distance_stats = np.empty([nb_boot, 2])
        for k in range(nb_boot):                        
            chromagram_boot_act = chromagram_act[boot_idx_act[k]:boot_idx_act[k]+nb_frames_dtw_act,:]
            chromagram_boot_est = chromagram_sample[boot_idx_est[k]:boot_idx_est[k]+nb_frames_dtw_est,:]
            distance_boot = cdist(chromagram_boot_act, chromagram_boot_est)
            distance_dtw_boot = dtw_fast(np.array(distance_boot, dtype=np.float64), self.dtw.weights_mul, 1, self.filter.max_consecutive_steps_real)
            distance_dtw_boot = distance_dtw_boot[np.isfinite(distance_dtw_boot[:,-1]), -1]
            distance_stats[k,0] = np.mean(distance_dtw_boot)
            distance_stats[k,1] = np.min(distance_dtw_boot)
                           
        a = 1
        
    def get_chromagram(self, wd, filename):
        '''
        The function can take as input a 'wav' or a '.mid' file.
        in the latter case, we have to generate the corresponding '.wav'.  
                
        Get the chromagram from the disk or process the '.wav' file and write 
        the chromagram to disk if need be.
        '''
        
        # Build a specific key for the chromagram in case we want to store.
        suffix_filename = "_chromagram_S{}_H{}_fcn{}_mode{}_.npy".format(self.sr, 
                                                                         self.hop_length,
                                                                         self.compute_chromagram_fcn.__name__, 
                                                                         self.chromagram_mode)                                                                                                                                                                                                             
        
        filename_chomagram = wd + utils.unmake_file_format(filename, self.file_extension) + suffix_filename
        

        # We check if the chromagram exists on disk, otherwise we generate it.         
        if os.path.isfile(filename_chomagram) and self.store_chromagram:
            chromagram_act = np.load(filename_chomagram)
        else:
            # If the input file is a '.mid', we have to generate the '.wav'
            # Also, store the midi object, it will be useful to get the position in ticks
            if self.file_extension == '.mid':
                self.midi_obj = pretty_midi.PrettyMIDI(wd + filename)
                
                # We force the program number for all instrument to be 0 (Grand piano acoustique)
                # This is a bit hacky, but the soundfont is only good for that programme.
                for instrument in self.midi_obj.instruments:
                    instrument.program = 0
                    
                # Make sure that we have the right format                                
                filename = utils.change_file_format(filename, '.mid', '.wav')
                
                # Synthesise the wav file with fluidsynth
                audio_data = self.midi_obj.fluidsynth(self.sr, start_new_process32=utils.fluidsynth_start_new_process32())
                
                # Write the wav
                utils.write_wav(wd + filename, audio_data, rate = self.sr)
                                                                        
            # Now we are sure to have a '.wav', we can retrieve it.
            audio_data_wav = lb.core.load(wd + utils.make_file_format(filename, '.wav'), sr = self.sr)
            
            # Turn into chromagram
            chromagram_act = self.compute_chromagram(audio_data_wav[0])
            
            # Store if need be.                                                                                                                                                                                         
            if self.store_chromagram:                                                 
                np.save(filename_chomagram, self.chromagram_act)            

        return(chromagram_act)
        
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
                                                 sr=self.sr,
                                                 hop_length=self.hop_length,
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
            
    
    def update_audio_data(self, new_audio_data):
        '''
        Update the buffer of audio data. We use this buffer to compute the live chromagram.
        '''
        
        # Check that we pass in the right type of data
        if new_audio_data.dtype is not np.dtype(utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0]):
            raise(TypeError('Input audio data need to be {}!'.format(utils.AUDIO_FORMAT_DEFAULT)))
        
        
        # Make sure that the chunk size is a multiple of hop size
        if (len(new_audio_data) % self.hop_length) != 0:
            raise ValueError('Chunk size need to be a multiple of hop size')
        
        # Append the new data
        self.audio_data = np.append(self.audio_data, new_audio_data)        
        self.audio_data_all = np.append(self.audio_data_all, new_audio_data)
        
        # Trim. There are two possible cases:
        # - we added a lot of data , i.e. len(new_audio_data) > self.min_len_sample: we want to keep all the new added data
        # - we added a an online chunk of data , i.e. len(new_audio_data) < self.min_len_sample: in this case, we only keep 
        # the most recent samples. 
        self.audio_data = self.audio_data[max(len(self.audio_data) - max(self.min_len_sample, len(new_audio_data)), 0):len(self.audio_data)]                        
                    
    def plot_chromagrams(self):
        """
        Plot both the estimated and the actual chromagram for comparison.
        """
        
        axarr = plt.subplots(2, 1)[1]
        
        # Plot chromagram_act
        utils.plot_chromagram(self.chromagram_act, sr=self.sr, hop_length=self.hop_length, ax=axarr[0], xticks_sec=True)
        axarr[0].set_title('Chromagram act')
        
        # Plot chromagram_est (up to idx_est). Will need to amend to sr_est and hop_length_est at some point...
        utils.plot_chromagram(self.chromagram_est[0:self.idx_est,:], sr=self.sr, hop_length=self.hop_length, ax=axarr[1], xticks_sec=True)
        axarr[1].set_title('Chromagram est')  
           
        
    def match_batch_online(self, audio_data_est):
        '''
        Align a target raw audio.
        The raw audio needs to be sampled as per the matcher's sample rate.
        Though the process is offline because we pass in the entire audio set, the 
        actual alignment does not use any forward-looking data.
         
        Return a mapping between the times of the target audio vs times of the "true" audio.  
        '''
        
        # Truncate the data to get a multiple of hop_length, we can afford to remove a 
        # bit of the data in the batch process.
        nb_obs = len(audio_data_est)
        nb_obs_dropped = nb_obs % self.hop_length    
        audio_data_est = audio_data_est[0:-nb_obs_dropped] if nb_obs_dropped else audio_data_est
                
        self.match_online(audio_data_est)
        
        # Get the alignment output
        # Not sure whether we should add/take out one frame... Also, we should use matcher_tmp.sr_est once possible
        times_cor_est = np.arange(len(self.positions_sec)) * self.hop_length / float(self.sr) 
        times_ori_est = np.array(self.positions_sec)
         
        return(times_cor_est, times_ori_est)
    
    def match_batch_offline(self, audio_data_est):
        '''
        Use the librosa DTW implementation to perform offline matching.
        '''
        
        # Compute the chromagram, use the engine of the matcher to ensure that both chromagrams have 
        # been computed with the same procedure.            
        chromagram_est = self.compute_chromagram(audio_data_est)
        
        # Set up the diagonal cost        
        weights_mul = np.array([1.0*self.dtw.diag_cost, 1.0, 1.0])   
        
        # Run the DTW alignment
        [cum_distance, best_path] = lb.core.dtw(self.dtw.features_act.T, chromagram_est.T, weights_mul=weights_mul)
        
        # We need to append the indices to go to (0,0).
        # Librosa DTW does not backtrack all the way to the origin. 
        # (early stop for compatibility with subsequence alignment)
        idx_est_first = best_path[-1, 1]
        best_path_add = np.hstack((np.zeros((idx_est_first, 1), dtype='int32'), np.atleast_2d(np.flip(np.arange(idx_est_first),0)).T))
        best_path = np.vstack((best_path, best_path_add))                         
        
        # Get the alignment output, reverse the order of the paths
        times_ori_est = np.flip(best_path[:,0], 0) * self.hop_length / float(self.sr)         
        times_cor_est = np.flip(best_path[:,1], 0) * self.hop_length / float(self.sr)
                            
        return(times_cor_est, times_ori_est, {'cum_distance':cum_distance, 'best_path':best_path}) 
                                                                                                
    def match_online(self, new_audio_data_est):
        '''
        Run the online matching procedure for a chunk of audio data.
        '''
        
        # Add the new data to the buffer
        self.update_audio_data(new_audio_data_est)
        
        # Compute the chromagram for the entire buffer. 
        # This is not computationally optimal, (since len(new_audio_data_est) << len(self.audio_data))
        # but it is safer. We may need to reduce the buffer size if this computation is too expensive.       
        chromagram_est = self.compute_chromagram(self.audio_data)                                                               
        
        # Only run the matching over the last (len(new_audio_data)/self.hop_length) segments of the chromagram        
        idx_frames = np.arange(chromagram_est.shape[0] - int(len(new_audio_data_est) / self.hop_length), chromagram_est.shape[0])
        
        # Run the online matching procedure, one frame at a time
        for n in idx_frames:
                        
            self.dtw.main_dtw(chromagram_est[n,:], 
                              self.filter.idx_act_lb, 
                              self.filter.idx_act_ub,
                              self.filter.max_consecutive_steps)
            
            self.filter.main_filter(self.dtw.cum_distance)
            
            # Propagate the best estimate of the position to the matcher itself,
            # and convert to seconds
            self.positions.append(self.filter.positions_base[-1])            
            self.positions_sec.append(self.positions[-1] * self.hop_length / float(self.sr))
            
            # If the input file was a '.mid' in the first place, convert the 
            # position to ticks
            if self.file_extension == '.mid':
                self.position_tick = self.midi_obj.time_to_tick(self.positions_sec[-1])   
