import pandas as pd
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache' # Enable librosa cache
import librosa as lb
import utils_audio_transcript as utils 
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# We set the window type to Hamming to avoid numerical
# issues while running the algorithm online 
# (the Hamming window does not go to zero at the edges)
WINDOW_TYPE = 'hamming'

class NoiseReducer():
    
    def __init__(self,
                 alpha_power_spectrum=0.99,
                 noise_bias_correction=1.5,
                 alpha_snr=0.98):
        
        # The smoothing for the power spectrum (in the estimation
        # of the noise
        self.alpha_power_spectrum = alpha_power_spectrum
        
        # The correction applied in the estimation of the noise
        # (the estimation is biased because we take the min)
        self.noise_bias_correction = noise_bias_correction
        
        # The smoothing of the prior SNR for the Ephraim-Malah procedure 
        self.alpha_snr = alpha_snr
         
        # Initialise the audio data buffers (input and output)
        self.audio_data = []
        self.audio_data_denoised = []
                
        self.n_fft = 1024
        self.hop_length = 512
        self.n_coef_fft = self.n_fft//2 + 1
        self.stft = np.zeros([0, self.n_coef_fft], dtype=np.complex64) + np.NaN
        # Store the magnitude and phase (redundant with STFT, could be removed)
        self.stft_mag = np.zeros(self.stft.shape) + np.NaN
        self.stft_phase = np.zeros(self.stft.shape, dtype=np.complex64) + np.NaN                
        
        # The (total) power spectrum smoothed (in the time dimension)
        self.smooth_power_spectrum = np.zeros(self.stft.shape) + np.NaN
        
        # The estimate for the noise power spectrum
        self.noise_estimate = np.zeros(self.stft.shape) + np.NaN
        
        # The gain (i.e. the frequency filter that we apply to the raw signal)
        self.gain = np.zeros(self.stft.shape) + np.NaN
        
        # Keep the post-cleaning STFT (only for reporting)
        self.stft_denoised = np.zeros(self.stft.shape) + np.NaN
        
        # After iterating the main loop, we have processed up to
        # (and including) self.idx_curr
        self.idx_curr = -1
        self.idx_prev = np.nan
        
        # Store the SNR (posterior and prior) for the Ephraim-Malah algorithm
        self.snr_prior = np.zeros(self.stft.shape) + np.NaN
        self.snr_post = np.zeros(self.stft.shape) + np.NaN
        
    def calc_online_stft(self, audio_data_new_length):
        '''
        Calculate the STFT online. 
        i.e. find how much of the previous audio data we need to take, append the new audio data
        such that the windowing is valid and compute the FFT. 
        '''
        # We need to get (n_fft - hop_length) samples from the previous audio data
        start_idx = max(len(self.audio_data) - audio_data_new_length - self.n_fft + self.hop_length, 0)
        
        # Calculate the STFT for the new blocks
        stft = lb.spectrum.stft(np.array(self.audio_data[start_idx:]), self.n_fft, self.hop_length, window=WINDOW_TYPE, center=False)
        
        # Also calculate the magnitude and phase sprectra
        [stft_mag, stft_phase] = lb.core.magphase(stft)   
        
        # Append the new blocks   
        self.stft = np.vstack((self.stft, stft.T))
        self.stft_mag = np.vstack((self.stft_mag, stft_mag.T))
        self.stft_phase = np.vstack((self.stft_phase, stft_phase.T))
        
        self.idx_prev = self.idx_curr 
        self.idx_curr = self.stft.shape[0]-1
        
    def calc_smooth_power_spectrum(self):
        '''
        Calculate the smoothed power spectrum
        '''            
        
        # For the first frame, we only have the raw power spectrogram
        idx_prev_adj = self.idx_prev
        if self.idx_prev < 0:
            self.smooth_power_spectrum = self.stft_mag[0,:]**2
            idx_prev_adj = idx_prev_adj + 1
        
        # After the first frame, we can smooth with EWMA update
        self.smooth_power_spectrum = np.vstack((self.smooth_power_spectrum, np.zeros([self.idx_curr - idx_prev_adj, self.n_coef_fft])))    
        for k in np.arange(idx_prev_adj + 1, self.idx_curr + 1):
            update = (1-self.alpha_power_spectrum) * self.stft_mag[k,:]**2
            self.smooth_power_spectrum[k, :] = self.alpha_power_spectrum * self.smooth_power_spectrum[k-1, :] + update
        
    
    def calc_noise_estimate(self):
        '''
        Calculate the noise estimate based on the running minimum of 
        the smoothed power spectrum.
        '''        
        min_n_frames_noise_estimation = 50
        n_frames_noise_estimation = 50
        
        # Until we have enough data for estimation, assume no noise
        if self.idx_curr < min_n_frames_noise_estimation:
            self.noise_estimate = np.vstack((self.noise_estimate, np.zeros([self.idx_curr - self.idx_prev, self.n_coef_fft])))
            return
        
        # Calculate the rolling minimum of the smoothed total power
        smooth_power_spectrum_new = self.smooth_power_spectrum[max(self.idx_prev - n_frames_noise_estimation, 0):, :] 
        noise_estimate_new = pd.DataFrame(smooth_power_spectrum_new).rolling(n_frames_noise_estimation).min().as_matrix()            
        noise_estimate_new = noise_estimate_new[len(noise_estimate_new) - (self.idx_curr - self.idx_prev):, :]
        
        # Zero-out nans (as the rolling min returns NaN in the initialisation period)        
        np.nan_to_num (noise_estimate_new, copy=False)                 
        
        # Floor the noise power to the total power 
        noise_estimate_new = np.minimum(noise_estimate_new, self.stft_mag[self.idx_prev+1:self.idx_curr+1 ,:]**2)            

        # Apply correction for bias (should go before the previous operation maybe?)
        noise_estimate_new = noise_estimate_new * self.noise_bias_correction
                
        # Append the new block 
        self.noise_estimate = np.vstack((self.noise_estimate, noise_estimate_new))
        
    def calc_gain_wiener(self):
        '''
        Calculate the Wiener filter
        Can be issued as an alternative to the Ephraim Malah gain calculation.  
        '''            
        power_total = self.stft_mag[self.idx_prev + 1:self.idx_curr + 1, :]**2        
        power_noise_estimate = self.noise_estimate[self.idx_prev + 1:self.idx_curr + 1, :]        
        
        gain = np.maximum(1 - power_noise_estimate/power_total, 0) 
        self.gain = np.vstack((self.gain, gain))
        
    def calc_gain_ephraim_malah(self):
        '''
        Refs:
        [1] Efficient Alternatives to the Ephraim and Malah Suppression 
        Rule for Audio Signal Enhancement, Wolfe P., Godsill S., 2003.
        [2] Single Channel Noise Reduction for Hands Free Operation 
        in Automotive Environments, Schmitt S., Sandrock M. and Cronemeyer, J., 2002.        
        '''        
        
        # Place-holders for SNRs and gain 
        snr_prior = np.zeros([self.idx_curr - self.idx_prev, self.n_coef_fft]) + np.NaN
        snr_post = np.zeros([self.idx_curr - self.idx_prev, self.n_coef_fft]) + np.NaN
        gain = np.zeros([self.idx_curr - self.idx_prev, self.n_coef_fft]) + np.NaN
        
        for n in range(self.idx_prev + 1, self.idx_curr + 1):
            
            k = n - self.idx_prev - 1
            
            # Floor the noise_estimate to 0+tol, as we need to divide Inf by Inf  
            noise_estimate = np.maximum(self.noise_estimate[n,:], np.finfo(np.float).eps) 
            snr_post[k,:] = self.stft_mag[n,:]**2 / noise_estimate # -1 needed??
            snr_post_floored = np.maximum(snr_post[k,:], 0.0) # Flooring needed?
            
            # Calculate the SNR prior in a "decision-directed" approach (see [2])
            if n == 0:
                snr_prior_raw = snr_post_floored
            else:
                noise_estimate_prev = np.maximum(self.noise_estimate[n-1,:], np.finfo(np.float).eps)
                gain_prev = self.gain[n-1, :] if k == 0 else gain[k-1, :]             
                snr_prior_raw = (gain_prev * self.stft_mag[n-1,:])**2 / noise_estimate_prev
                
            snr_prior[k,:] = self.alpha_snr*snr_prior_raw + (1-self.alpha_snr)*np.maximum(snr_post[k,:]-1, 0.0)
            
            # Ephraim-Malah approximation by Wolfe 
            # (Minimum mean square error spectral power estimator in [1])
            p = snr_prior[k,:]/(1+snr_prior[k,:])            
            gain[k,:] = np.sqrt(p * (1/snr_post[k,:] + p))   
            
        # Append the gain and SNRs for the new block 
        self.snr_prior = np.vstack((self.snr_prior, snr_prior))           
        self.snr_post = np.vstack((self.snr_post, snr_post))
        self.gain = np.vstack((self.gain, gain))
               
        
    def reconstruct_audio_data(self): 
        '''        
        Apply gain.
        Inverse FFT to reconstruct the denoised audio data.
        '''
        
        # Only apply on the newly processed chunks
        idxs = np.arange(self.idx_prev + 1, self.idx_curr + 1)
        
        # denoised = gain X magnitude total X phase total
        stft_denoised = self.gain[idxs, :] * self.stft_mag[idxs, :] * self.stft_phase[idxs, :]
        self.stft_denoised = np.vstack((self.stft_denoised, stft_denoised)) 
        
        # Reconstruct the signal in the time space    
        audio_data_denoised = lb.spectrum.istft(stft_denoised.T, self.hop_length, window=WINDOW_TYPE, center=False).tolist()
        
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
        self.calc_gain_ephraim_malah()
        self.reconstruct_audio_data()
        

                
wd = utils.WD + "Samples\SaarlandMusicData\SaarlandMusicDataRecorded//"
filename_wav = wd + "Ravel_JeuxDEau_008_20110315-SMD.wav" #"Chopin_Op066_006_20100611-SMD.wav"
audio_data = (lb.core.load(filename_wav, sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]).astype(np.float64)

noise_reducer = NoiseReducer()
# for k in np.arange(len(audio_data)//1024):    
for k in np.arange(500):
    noise_reducer.main(audio_data[k*1024:k*1024 + 1024])#[-220160:]
    
# noise_reducer2 = NoiseReducer()
# noise_reducer2.main(audio_data)
# 
# figure()
# plt.plot(noise_reducer.smooth_power_spectrum[355:,146])
# plt.plot(noise_reducer.stft_mag[355:,146]**2)
# plt.plot(noise_reducer.snr_post[355:,146])
# plt.plot(noise_reducer.snr_prior[355:,146])
# plt.plot(noise_reducer.gain[355:,146])
# 
# figure()
# plt.plot(noise_reducer.audio_data)
# plt.plot(noise_reducer.audio_data_denoised)
# 
#     
# utils.write_wav(wd + "Ravel_JeuxDEau_008_20110315-SMD_denoised.wav", np.array(noise_reducer.audio_data_denoised), rate=utils.SR)



        