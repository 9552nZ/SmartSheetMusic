'''
The script shows how to use the MatcherEvaluation class.
As an example, we run a series of corruption configs to test the Matcher class. 
'''
import corrupt_midi
import utils_audio_transcript as utils
import librosa as lb
import numpy as np
from matcher_evaluation import MatcherEvaluator
from score_following import Matcher

# Set up the series of corruption we want to apply
configs_corrupt = [
    {'change_inst_prob':1.0},
    {'warp_func':corrupt_midi.warp_sine, 'warp_func_args':{'nb_wave' : None}},
    {'warp_func':corrupt_midi.warp_linear, 'warp_func_args':{'multiplier' : 0.8}},                                
    {'velocity_std':0.5},
]

# Set up working directory. The midi file should be saved there
wd = utils.WD_MATCHER_EVALUATION 
filename_midi = "Chopin_Op028-01_003_20100611-SMD.mid"

# Set up the config that we use for the Matcher
config_matcher = {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':np.inf, 'n_chroma':84}}

# Run the evaluation procedure
matcher_evaluator = MatcherEvaluator(Matcher, wd, filename_midi, config_matcher=config_matcher, configs_corrupt=configs_corrupt)
matcher_evaluator.main_evaluation()

# These are all the alignment stats
alignement_stats = matcher_evaluator.alignement_stats

# Plot the alignment error
matcher_evaluator.plot_alignment_error()