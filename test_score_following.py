import matplotlib.pyplot as plt
import score_following
import ScoreClass
import corrupt_midi
import pretty_midi
import create_data
import fluidsynth
import librosa as lb
import numpy as np
import utils_audio_transcript as utils

        
# midi_object = pretty_midi.PrettyMIDI(wd + filename1)
# matcher = MatcherMidi(wd, filename1, filename2)
# matcher.main_matcher()
# plt.plot(matcher.path[0], matcher.path[1])
# fig = plt.figure()
# fig.add_subplot(2,1,1)
# plt.imshow(midiread(wd + filename1, r=(0, 127)).piano_roll.T, origin='lower', aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r)
# fig.add_subplot(2,1,2)
# plt.imshow(midiread(wd + filename2, r=(0, 127)).piano_roll.T, origin='lower', aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r)
# utils.synthetise(wd + filename_midi1, wd + filename_wav1)


# wd = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\Chopin_op28_3/"
#  
# filename_midi1 = "Chopin_Op028-01_003_20100611-SMD.mid"
# filename_midi2 = "Chopin_Op028-01_003_20100611-SMD_corrupt.mid"
# filename_wav1 = "Chopin_Op028-01_003_20100611-SMD.wav"
# filename_wav2 = "Chopin_Op028-01_003_20100611-SMD_corrupt.wav"

# midi_object = pretty_midi.PrettyMIDI(wd + filename_midi1)
# original_times = np.linspace(0.0, midi_object._PrettyMIDI__tick_to_time[-1], int(midi_object._PrettyMIDI__tick_to_time[-1]/0.05))
# adjusted_times,  diagnostics = corrupt_midi.corrupt_midi(midi_object, original_times)        
# midi_object.write(wd + filename_midi2)

# utils.write_wav(wd + filename_wav1, pretty_midi.PrettyMIDI(wd + filename_midi1).fluidsynth(44100), rate = 44100)
# utils.write_wav(wd + filename_wav2, pretty_midi.PrettyMIDI(wd + filename_midi2).fluidsynth(44100), rate = 44100)

# SR = 11025
# N_FFT = 2048
# HOP_LENGTH = 1024
# 
# win_len_smooth = 1
# tuning = 0.0
# chromagram_act = lb.feature.chroma_cens(y=audio_data_act, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T
# audio_data_est = lb.core.load(wd + filename_wav2, sr = SR, offset=0.0, duration=None)[0]
# chromagram_est = lb.feature.chroma_cens(y=audio_data_est, win_len_smooth=win_len_smooth, sr=SR, hop_length=HOP_LENGTH, chroma_mode='stft', n_fft=N_FFT, tuning=tuning).T
# np.save( wd + 'chromagram_act.npy', chromagram_act)
# np.save( wd + 'chromagram_est.npy', chromagram_est)
# chromagram_act = np.load( wd + 'chromagram_act.npy')
# chromagram_est = np.load( wd + 'chromagram_est.npy')

# nb_frames_est = chromagram_est.shape[0]
# 
# matcher = score_following.Matcher(wd, filename_wav1, SR, N_FFT, HOP_LENGTH)
# 
# for n in range(nb_frames_est):
#     matcher.main_matcher(chromagram_est[n,:])
#     print n, matcher.position_sec 
# 
# plt.plot(matcher.best_paths[-1].rows, matcher.best_paths[-1].cols)


# wd = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\Chopin_op28_3/"
# filename_midi = "Chopin_Op028-01_003_20100611-SMD.mid"

# wd = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\Bach_BWV888_02_008/"
# filename_midi = "Bach_BWV888-02_008_20110315-SMD.mid"

wd = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\Beethoven_Op027No1/"
filename_midi = "Beethoven_Op027No1-03_003_20090916-SMD.mid"


matcher_evaluator = score_following.MatcherEvaluator(wd, filename_midi)
# matcher_evaluator.filenames_wav_corrupt = ['Chopin_Op028-01_003_20100611-SMD.wav']
# matcher_evaluator.times_cor_act_all = [matcher_evaluator.times_ori_act] 
# matcher_evaluator.align()
# matcher_evaluator.evaluate()

matcher_evaluator.main_evaluation()
matcher_evaluator.plot_alignment_error()
# plt.plot(matcher_evaluator.alignement_stats[-1])
utils.plot_chromagram(matcher_evaluator.matchers[0].chromagram_est[0:matcher_evaluator.matchers[0].idx_est,:])

a = 1