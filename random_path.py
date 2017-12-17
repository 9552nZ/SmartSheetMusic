import utils_audio_transcript as utils
import pandas as pd
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

def get_features_base(nb_sample, nb_feature, norm):
    '''
    Using the .wav samples from the AUDIOSET dataset, compute some features.
    We pick the first few files (and get something random in essence).    
    Return a features matrix of size (nb_sample X nb_feature).
    '''
    filename_info = utils.WD_AUDIOSET + "\VerifiedDataset\\VerifiedDatasetRecorded\\" + "info.pkl"
    df_audioset = pd.read_pickle(filename_info)
    sr = utils.SR
    
    features = np.zeros((nb_sample, nb_feature))+np.nan
    cnt_samples = 0
    for _, row in df_audioset.iterrows():
        if row["classification"] == 1:
            # Load the sample  
            audio_data_wav = lb.core.load(row["filename_wav"], sr = sr, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]        
            
            # Compute the features
            features_tmp = lb.feature.chroma_stft(audio_data_wav, sr=sr, n_fft=utils.N_FFT, hop_length=utils.HOP_LENGTH, n_chroma=nb_feature, tuning=0.0, norm=norm) 
            nb_sample_new = features_tmp.shape[1]
            
            # Extract just as many featrues as we need
            features[cnt_samples:cnt_samples+nb_sample_new, :] = features_tmp.T[0:nb_sample-cnt_samples, :]
            
            cnt_samples += nb_sample_new
            
            if cnt_samples > nb_sample:
                break 
            
    return(features)

def generate_random_duplicates(idxs, proba_duplicate, mean_nb_repetitions):
    '''
    Given a array of integers as input (e.g. the index for an array), 
    randomly duplicate some of the elements and return the random array of integers. 
    The ordering of the array remains stable through the operation.
    
    The number of repetitions are sampled from an exponential distribution with 
    mean mean_nb_repetitions.
    '''
    
    nb_sample = len(idxs)
    
    # Generate random binary flags for duplicates
    duplicate = np.random.uniform(0, 1, size=nb_sample) < proba_duplicate 
    
    # For each duplicate, draw the number of repetitions from an exponential distribution
    nb_duplicate = sum(duplicate)
    repetitions = np.random.exponential(scale=mean_nb_repetitions-1, size=nb_duplicate)+1
    repetitions = np.array(np.round(repetitions), dtype=np.int)
    
    # Remap    
    repetitions_all = np.ones(nb_sample, dtype=np.int)
    repetitions_all[duplicate] = repetitions        
    idxs_new = np.array([idxs[x] for x in idxs for _ in range(repetitions_all[x])], dtype=np.int)
    
    return(idxs_new)

def generate_random_deletions(idxs, proba_deletion, mean_nb_deletions):    
    '''
    Given a array of integers as input (e.g. the index for an array), 
    randomly remove some of the elements and return the random array of integers. 
    The ordering of the array remains stable through the operation.
    We delete consecutive blocks of the array. 
    
    The number proba_deletion is the probability re. item i is the proabiliby that a 
    deleted block starts in i (this is NOT the probability that item i gets deleted). 
    
    The number of deletions are sampled from an exponential distribution with 
    mean mean_nb_deletions.

    '''
        
    nb_sample = len(idxs)
    
    # Generate random binary flags for duplicates
    deleted = np.random.uniform(0, 1, size=nb_sample) < proba_deletion 
    
    # For each duplicate, draw the number of repetitions from an exponential distribution
    nb_deletions = sum(deleted)
    deletions = np.random.exponential(scale=mean_nb_deletions-1, size=nb_deletions)+1
    deletions = np.array(np.round(deletions), dtype=np.int)
    
    # Remap    
    deletions_all = np.zeros(nb_sample, dtype=np.int)
    deletions_all[deleted] = deletions        
    mask = np.ones(nb_sample, dtype=bool)
    for k in range(nb_sample):            
        mask[k:k+deletions_all[k]] = False
        
    idxs_new = idxs[mask]    
    
    return(idxs_new)    

def add_random_noise(features, std_dev_noise, norm):
    '''
    Add a gaussian exponential multiplicative noise to the features
    (keeps positivity). 
    Rescale the features such that the maximum is 1 (per row).
    '''
    
    # Generate the noise
    noise = np.random.normal(0.0, scale=std_dev_noise, size=features.shape)
    noise = np.exp(noise)
    
    # Add it to the features 
    features *= noise
    
    # Rescale the features
    features = lb.util.normalize(features, norm=norm, axis=0)
    
    return(features)    
    

class RandomPathGenerator():
    def __init__(self, nb_feature=12, nb_sample=300):
        
        norm = np.inf
        features_base = get_features_base(nb_sample, nb_feature, norm)
        idxs_base = np.arange(nb_sample)
        idxs_new = generate_random_duplicates(idxs_base, 0.1, 2)
        idxs_new = generate_random_deletions(idxs_new, 0.01, 5)
        
        features_new = features_base[idxs_new, :] 
        features_new = add_random_noise(features_new, 0.0, norm)
        


cum_distance_dtw, wp_dtw = lb.dtw(features_base.T, features_new.T)

plt.figure() 
plt.plot(wp_dtw[:,1], wp_dtw[:,0])
plt.plot(idxs_new)