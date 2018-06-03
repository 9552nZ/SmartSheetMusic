import pandas as pd
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache' # Enable librosa cache
import librosa as lb
import utils_audio_transcript as utils 
from time import sleep
import matplotlib.pyplot as plt

class NoiseReducer():
    
    def __init__(self):
        
        # Initialise the audio data buffers (input and output)
        self.audio_data = []
        self.audio_data_denoised = []
                
        self.n_fft = 1024
        self.hop_length = 512
        self.n_coef_fft = self.n_fft//2 + 1
        self.stft = np.zeros([0, self.n_coef_fft], dtype=np.complex64) + np.NaN
        
        # The (total) power spectrum smoothed (in the time dimension)
        self.smooth_power_spectrum = np.zeros(self.stft.shape) + np.NaN
        
        # The estimate for the noise power spectrum
        self.noise_estimate = np.zeros(self.stft.shape) + np.NaN
        
        # The gain (i.e. the frequency filter that we apply to the raw signal)
        self.gain = np.zeros(self.stft.shape) + np.NaN
        
        # After iterating the main loop, we have processed up to
        # (and including) self.idx_curr
        self.idx_curr = -1
        self.idx_prev = np.nan

    @staticmethod
    def stft_to_power_spectrum(s):
        '''
        Convert a stft to a power spectrum
        '''
        return np.abs(s)**2
        
    def calc_online_stft(self, audio_data_new_length):
        
        start_idx = max(len(self.audio_data) - audio_data_new_length - self.n_fft + self.hop_length, 0)
          
        stft_new = lb.spectrum.stft(np.array(self.audio_data[start_idx:]), self.n_fft, self.hop_length, center=False)        
        self.stft = np.vstack((self.stft, stft_new.T))
        
        self.idx_prev = self.idx_curr 
        self.idx_curr = self.stft.shape[0]-1
        
    def calc_smooth_power_spectrum(self):
        '''
        Calculate the smoothed power spectrum
        '''
        alpha = 0.95        
        
        # For the first frame, we only have the raw power spectrogram
        idx_prev_adj = self.idx_prev
        if self.idx_prev < 0:
            self.smooth_power_spectrum = self.stft_to_power_spectrum(self.stft[0,:])
            idx_prev_adj = idx_prev_adj + 1
        
        # After the first frame, we can smooth with EWMA update
        self.smooth_power_spectrum = np.vstack((self.smooth_power_spectrum, np.zeros([self.idx_curr - idx_prev_adj, self.n_coef_fft])))    
        for k in np.arange(idx_prev_adj + 1, self.idx_curr + 1):
            self.smooth_power_spectrum[k, :] = alpha * self.smooth_power_spectrum[k-1, :] + (1-alpha) * self.stft_to_power_spectrum(self.stft[k,:])
    
    def calc_noise_estimate(self):
        '''
        Calculate the noise estimate based on the running minimum of 
        the smoothed power spectrum.
        '''
        min_n_frames_noise_estimation = 50
        n_frames_noise_estimation = 50
        
        if self.idx_curr < min_n_frames_noise_estimation:
            self.noise_estimate = np.vstack((self.noise_estimate, np.zeros([self.idx_curr - self.idx_prev, self.n_coef_fft])))
            return
        
        smooth_power_spectrum_new = self.smooth_power_spectrum[max(self.idx_prev - n_frames_noise_estimation, 0):, :] 
        noise_estimate_new = pd.DataFrame(smooth_power_spectrum_new).rolling(n_frames_noise_estimation).min().as_matrix()
        np.nan_to_num (noise_estimate_new, copy=False)
        start_idx = len(noise_estimate_new) - (self.idx_curr - self.idx_prev)
        self.noise_estimate = np.vstack((self.noise_estimate, noise_estimate_new[start_idx:, :]))
        
    def reconstruct_audio_data(self): 
        '''        
        Calculate the Wiener filter and apply.
        Inverse FFT to reconstruct the denoised audio data.
        '''
        # Get the most recent chunk of STFT and split into magitude and phase
        stft_total = self.stft[self.idx_prev + 1:self.idx_curr + 1, :]
        [mag_total, phase_total] = lb.core.magphase(stft_total)
        power_noise_estimate = self.noise_estimate[self.idx_prev + 1:self.idx_curr + 1, :]        
        
        gain = np.maximum(1 - power_noise_estimate/mag_total**2, 0) 
        self.gain = np.vstack((self.gain, gain))
        
        mag_signal = gain * mag_total
        
        stft_signal = mag_signal * phase_total 
            
        audio_data_denoised = lb.spectrum.istft(stft_signal.T, self.hop_length, center=False).tolist()
        
        # We need to ditch the beginning of the series (as it add been already 
        # been processed in the previous iterations).
        # This does not apply to the first frame
        if self.idx_prev < 0:
            self.audio_data_denoised.extend(audio_data_denoised)
        else:
            self.audio_data_denoised.extend(audio_data_denoised[self.n_fft - self.hop_length:])
        
    
    def main(self, audio_data_new):
        
        # Check that the input audio is as expected
        if len(audio_data_new) < self.n_fft or len(audio_data_new) % self.hop_length != 0:
            raise IndexError("Bad size for the new chunk of audio")
        
        self.audio_data.extend(audio_data_new)        
        self.calc_online_stft(len(audio_data_new))
        self.calc_smooth_power_spectrum()
        self.calc_noise_estimate()
        self.reconstruct_audio_data()
                
wd = utils.WD + "Samples\SaarlandMusicData\SaarlandMusicDataRecorded//"
filename_wav = wd + "Chopin_Op066_006_20100611-SMD.wav"
audio_data = lb.core.load(filename_wav, sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]

noise_reducer = NoiseReducer()
chunk =  int(len(audio_data)/noise_reducer.n_fft) * noise_reducer.n_fft

noise_reducer.main(audio_data[0:102400])

noise_reducer2 = NoiseReducer()
for k in np.arange(100):
    noise_reducer2.main(audio_data[1024*k:1024*k+1024])
utils.write_wav(wd + "Chopin_Op066_006_20100611-SMD_denoised2.wav", np.array(noise_reducer2.audio_data_denoised), rate=utils.SR)
    
utils.write_wav(wd + "Chopin_Op066_006_20100611-SMD_denoised.wav", np.array(noise_reducer.audio_data_denoised), rate=utils.SR)

    
a = 1
    
    
    







        