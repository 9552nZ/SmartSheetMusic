'''
This config stores some variables used for the matching and for 
the evaluation procedures. 
'''

import corrupt_midi
import librosa as lb
import numpy as np

# The list of configurations used to corrupt the .mid files
configs_corrupt = [
    {'change_inst_prob':1.0},
    {'warp_func':corrupt_midi.warp_sine, 'warp_func_args':{'nb_wave' : None}},
    {'warp_func':corrupt_midi.warp_sine, 'warp_func_args':{'nb_wave' : 5.0}},
    {'warp_func':corrupt_midi.warp_linear, 'warp_func_args':{'multiplier' : 0.8}},
    {'warp_func':corrupt_midi.warp_linear, 'warp_func_args':{'multiplier' : 1.25}},                                
    {'velocity_std':0.5},
    {'velocity_std':2.0},
#             {},
]
 
#  List all the midi files in the evaluation universe
filenames_mid = [ 
    "Bach_BWV871-02_002_20090916-SMD.mid",
    "Bach_BWV888-02_008_20110315-SMD.mid",
    "Bartok_SZ080-02_002_20110315-SMD.mid",
    "Beethoven_Op027No1-02_003_20090916-SMD.mid",
    "Chopin_Op028-01_003_20100611-SMD.mid",
]
# 
# # configs_matcher = []
# #     for diag_cost in np.arange(0.8, 2.2, 0.2):
# #         configs_matcher.append({'diag_cost':diag_cost})
# 
# List the concurrent configs for the matcher
configs_matcher = [
#     {'compute_chromagram_fcn': lb.feature.chroma_stft, 'compute_chromagram_fcn_kwargs':{'n_fft':2048}},                   
#     {'compute_chromagram_fcn': lb.feature.chroma_cens, 'compute_chromagram_fcn_kwargs':{'win_len_smooth':5}},
#     {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{ 'norm':np.inf}, 'chromagram_mode':0},
#     {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':2}, 'chromagram_mode':0},
    {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':2, 'n_chroma':84}},
    {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':np.inf, 'n_chroma':84}},
    {'compute_chromagram_fcn': lb.feature.chroma_stft, 'compute_chromagram_fcn_kwargs':{'norm':2, 'n_fft':2048, 'n_chroma':84}},    
#     {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':None}, 'chromagram_mode':0},
#     {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':None}, 'chromagram_mode':1},
#     {'compute_chromagram_fcn': lb.feature.chroma_cqt, 'compute_chromagram_fcn_kwargs':{'norm':None}, 'chromagram_mode':2},
#                         {'compute_chromagram_fcn': lb.feature.chroma_cens, 'compute_chromagram_fcn_kwargs':{'chroma_mode':'stft'}},                                                
]

# configs_corrupt = [{'velocity_std':2.0}]
# filenames_mid = ["Chopin_Op028-01_003_20100611-SMD.mid"]
# configs_matcher = [{'compute_chromagram_fcn': lb.feature.chroma_cens, 'compute_chromagram_fcn_kwargs':{'win_len_smooth':5}}]