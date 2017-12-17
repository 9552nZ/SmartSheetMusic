import librosa as lb
import numpy as np
import corrupt_midi
import utils_audio_transcript as utils
import pretty_midi
from math import floor, ceil
import copy
import matplotlib.pyplot as plt
import config


class MatcherEvaluator():
    """
    Class to evaluate the score following procedure.
    1) Apply several corruptions to the original Midi file
    2) Perform alignment for each
    3) Compute the difference between ground-truth and estimated positions
    """     
    def __init__(self, class_matcher, wd, filename_midi, config_matcher={}, configs_corrupt={}):
        
        self.wd = wd
        self.class_matcher = class_matcher
        self.filename_midi_act = filename_midi        
        self.midi_object = pretty_midi.PrettyMIDI(self.wd + self.filename_midi_act) # Keep the original Midi file in memory 
        self.len_act_sec = self.midi_object._PrettyMIDI__tick_to_time[-1]
        self.times_ori_act = np.linspace(0.0, self.len_act_sec, int(self.len_act_sec/0.01)) # The reference timestamps in the original data
        self.times_est_all = [] # Placeholder for the output times of the alignment procedure [corrupted, original]
        self.times_cor_act_all = [] # Placeholder for the corrupted times (ground-truth)
        self.alignement_stats = [] # Placeholder for the output of the evaluation procedure
        
        # We keep the below parameters here as they are part of the evaluation procedure
        # (they should not be passed in the matcher's config)  
        self.sr = utils.SR # Used for both synthesis of the Midi files and alignment
        self.hop_length = utils.HOP_LENGTH # Used by the Matcher, but also by the evaluation procedure
                 
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
            audio_data = midi_object_corrupt.fluidsynth(self.sr, start_new_process32=utils.fluidsynth_start_new_process32())
            
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
        
        # Initialise the matcher
        matcher = self.class_matcher(self.wd, self.filename_midi_act, self.sr, self.hop_length, **self.config_matcher)
        
        # Loop over the configs
        for filename_wav_corrupt in self.filenames_wav_corrupt:
            
            print('Aligning {}'.format(filename_wav_corrupt))
            
            # Copy the matcher as it will be modified
            matcher_tmp = copy.deepcopy(matcher)
            
            # Load the audio data
            audio_data_est = lb.core.load(self.wd + filename_wav_corrupt, sr=matcher_tmp.sr_act)[0]
            
            # Find the starting and ending rests
            rests = utils.find_start_end_rests(audio_data_est, matcher_tmp.sr_act) # No need to pass in hop_size and n_fft
            self.times_rests.append(rests)
            
            # Perform the offline alignment and append the results
            [times_cor_est, times_ori_est] = matcher_tmp.match_batch_online(audio_data_est)
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
    
    def plot_alignment_error(self, idx_matcher):
        '''
        Plot the alignment error for one the corruption config.
        '''
        
        utils.plot_alignment(self.times_est_all[idx_matcher][:,0], 
                             self.times_est_all[idx_matcher][:,1], 
                             self.times_cor_act_all[idx_matcher], 
                             self.times_ori_act,
                             times_ori_est_filtered=self.matchers[idx_matcher].positions_sec_filtered)
        
                    
    def plot_alignment_error_multi(self):
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