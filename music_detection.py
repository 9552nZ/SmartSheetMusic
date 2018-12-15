import pandas as pd
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache' # Enable librosa cache
import librosa as lb
import utils_audio_transcript as utils
from csv import reader
from pickle import load, dump, HIGHEST_PROTOCOL
from os.path import isfile 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from ntpath import basename
# from subprocess import Popen
from time import sleep
from shutil import copyfile
from keras.models import load_model
import tensorflow as tf


FILENAME_INFO = "info.pkl"

def get_config_features(sr, n_fft, hop_length, idxs=[0,1,2]):
    '''
    Get the list of all interesting features that can be used to fit the NN model.
    ''' 
    
    diff = lambda x: np.hstack((np.zeros((x.shape[0], 1)), np.diff(x)))
    pos_diff = lambda x: np.hstack((np.zeros((x.shape[0], 1)), np.maximum(np.diff(x),0)))
    var = lambda x: np.array([[np.var(x)]])
       
    config_features = [                                                                             
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},
                                               
        {'fcn': mfcc, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                        
        {'fcn': spectral_flux, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                         
        {'fcn': audio_level_fcn, 'normalise':False, 'level_normalise': False, 'spectrum_based':False, 
         'fcn_kwargs':{'hop_length':hop_length}},
                                                     
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':None, 'n_chroma':84, 'tuning':0.0}},
                                                                                                  
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':pos_diff, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},
                                                                                                          
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':12, 'tuning':0.0}},
                                                               
        {'fcn': lb.feature.spectral_centroid, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':1, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                        
        {'fcn': lb.feature.spectral_bandwidth, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':1,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm': True}},
                        
        {'fcn': lb.feature.spectral_rolloff, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':1,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                        
        {'fcn': lb.feature.zero_crossing_rate, 'normalise':False, 'level_normalise': True, 'spectrum_based':False,
         'fcn_kwargs':{'frame_length':n_fft, 'hop_length':hop_length}},
                                               
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':diff, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},
                                                                                                                                                                                       
        {'fcn': mfcc, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':diff,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                        
        {'fcn': mfcc, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':pos_diff,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
                        
        {'fcn': raw_audio_fcn, 'normalise':False, 'level_normalise': True, 'spectrum_based':False, 'fcn_kwargs':{}},
                                                                    
        {'fcn': check_is_silence, 'normalise':False, 'level_normalise': True, 'spectrum_based':False, 'fcn_kwargs':{}},
                                
        {'fcn': lb.feature.zero_crossing_rate, 'normalise':False, 'level_normalise': True, 'spectrum_based':False, 'post_process':var,
         'fcn_kwargs':{'frame_length':hop_length, 'hop_length':hop_length}},
                                                                   
        {'fcn': spectral_flux, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':var, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}}, 
  
        {'fcn': high_zero_crossing_rate, 'normalise':False, 'level_normalise': True, 'spectrum_based':False,  
         'fcn_kwargs':{'hop_length':hop_length}},                           
                       
        {'fcn': low_short_term_energy_ratio, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':1,  
        'fcn_kwargs':{'n_fft':n_fft, 'hop_length':hop_length}},
                       
        {'fcn': spectrogram_image_features, 'normalise':False, 'level_normalise': True, 'spectrum_based':True, 'spectrum_power':1, 
         'fcn_kwargs':{'n_fft':n_fft, 'hop_length':hop_length}},
                                                                                                                                                                    
        ] 
    
    # Only return the configs that works the best
    config_features = [config_features[k] for k in idxs]
    
    return config_features

def raw_audio_fcn(y):
    '''
    Wrapper to return the raw audio
    '''
    return y[np.newaxis,:]

def mfcc(y, S, sr, n_fft, hop_length):
    '''
    Emulate the librosa MFCC function to accept the pre-computed power spectrum
    Exclude the first coefficient as it is only representative of audio level (sum of energy).
    Normalise by 100 to have a distribution closer to the one of the chromagram 
    (calibrated empirically for sr=11025, n_fft=1024, hop_length=512) 
    '''
    if not (sr == 11025 and n_fft==1024 and hop_length==512):
        raise IOError('MFCC normalisation not calibrated for these parameters')
    
    S_new = lb.core.spectrum.power_to_db(lb.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length))
    mfcc_raw = lb.feature.mfcc(sr=sr, S=S_new, n_fft=n_fft, hop_length=hop_length)
    mfcc_raw = mfcc_raw[1:,:] # Exclude the first coefficient
    mfcc_raw /= 100
            
    return(mfcc_raw)

def check_is_silence(y, threshold=-50, func=np.max):
    '''
    Check if the input is below the silence threshold
    '''
    db = 20 * np.log10(func(y))
    is_silence = np.expand_dims(np.array([db < threshold], dtype=bool), axis=0)
        
    return(is_silence)

def audio_level_fcn(y, hop_length):
    '''
    Estimate the loudness level.
    '''    
    # Pre-allocate results of size len(y) / hop_length + 1
    # the +1 comes from the centering of the STFT calculation, and we align the 
    # index with the STFT here
    nb_sample = len(y)
    audio_level = np.zeros((1, nb_sample//hop_length + 1)) + np.nan
    
    # Calculate the loudness as the mean absolute value    
    for (k,), val in np.ndenumerate(np.arange(0, nb_sample + hop_length, hop_length)):        
        audio_level[0, k] = np.mean(np.abs(y[max(val-hop_length//2, 0):(val+hop_length//2)])) 
    
    return(audio_level)

def spectral_flux(y, S, sr, n_fft, hop_length):
    '''
    Calculate the spectral flux as an alternative feature
    '''
    # Normalise the power spectrum with L2 (only if S != 0)     
    divisor = np.linalg.norm(S, ord=2, axis=0, keepdims=True)
    mask = divisor[0,:] > 0.0
    S[:,mask] = S[:,mask] / divisor[:,mask]     
     
    # Compute the flux as the temporal change
    flux = np.concatenate([np.zeros(1), np.linalg.norm(np.diff(S), ord=2, axis=0)])
        
    # Return as 2D array
    return np.expand_dims(flux, axis=0)

def high_zero_crossing_rate(y, hop_length):
    '''
    Ratio of frames whom ZCR is greater than 1.5x the clip-average ZCR.
    See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.1742&rep=rep1&type=pdf
    '''     
    
    zcr = lb.feature.zero_crossing_rate(y, frame_length=hop_length, hop_length=hop_length)
    hzcrr = np.mean(zcr > 1.5 * np.mean(zcr), keepdims=True) 
    
    return(np.expand_dims(hzcrr, axis=0))

def low_short_term_energy_ratio(y, S, n_fft, hop_length):
    '''
    Ratio of frames whom STE is less than 0.5x the clip-average ZCR.
    See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.1742&rep=rep1&type=pdf
    
    The n_fft argument is not used to avoid overlapping the windows.
    '''  
        
    ste = np.square(lb.feature.rmse(S=S, frame_length=hop_length, hop_length=hop_length))
    lster = np.mean(ste < 0.5 * np.mean(ste), keepdims=True)
    
    return(lster)
    return(ste)

def spectrogram_image_features(y, S, n_fft, hop_length):
    '''
    Frequency down-sampled spectrogram.
    As defined in https://arxiv.org/pdf/1604.06338.pdf
    Also see: https://kar.kent.ac.uk/51341/1/machine_hearing.pdf
    '''
    
    # Number of buckets we want to get
    nb_bucket = 52 
    nb_sample_per_bucket = (S.shape[0] // nb_bucket) + 1

    # Proceed in 2 steps (we don't have a whole number of windows)    
    S_reshaped1 = S[0:nb_sample_per_bucket*(nb_bucket-1), :].reshape(nb_sample_per_bucket, (nb_bucket-1), S.shape[1])
    S_reshaped2 = S[nb_sample_per_bucket*(nb_bucket-1):, :]
    
    # Calculate the mean over the buckets
    S_means = np.vstack((np.mean(S_reshaped1, axis=0), np.mean(S_reshaped2, axis=0, keepdims=True)))
    
    # Denoise (for each frequency bucket, remove the time-minimum)
    sif = S_means - np.min(S_means, axis=1, keepdims=True)
   
    return(sif)

def build_df_audio_data(data_type):
    """
    Build the dataframe with the list of audio samples.
    data_type can either be "test" or "train" 
    """
    
    from os import listdir
    from os.path import splitext
    
    labels = [0, 1]
    df = []
    for k in labels:
        root = utils.WD + "Samples\\{}\\{}\\".format(data_type, k)
        files = listdir(root)
        files = [f for f in files if splitext(f)[1] == ".wav"]
        df.append(pd.DataFrame({'classification':np.zeros(len(files), dtype=np.int32) + k, 
                                'filename_wav':[root + f for f in files], 
                                'valid':np.ones(len(files), dtype=bool)}))
    df = pd.concat(df)
    
    return(df)

def add_noise(wav, snr):
    '''
    wav is the (uncorrupted) signal.
    snr is the desired signal to noise ratio (expressed as the average power 
    of the signal divided by the average power of the noise).  
    '''    
    power = np.mean(wav**2)
    noise = np.sqrt(power/snr) * np.random.randn(len(wav))
    
    return(wav + noise)
    
def main_add_noise(wd, sr):
    '''
    Top-level API to add noise to multiple tracks.
    The objective was to match the difference between a good-quality recording vs. 
    a home recording with a poor quality mic.    
    '''
    # The directory to store the noisy samples
    wd_noisy = wd + 'VerifiedDatasetNoiseAdded\\'
    
    # Read the dataframe with all the samples info
    filename_df_audioset = FILENAME_INFO
    df_audioset = pd.read_pickle(wd + filename_df_audioset)
    df_audioset_noisy = df_audioset 
    
    # Set up the target signal-to-noise ratio 
    snr_db = 15.0
    snr_pow = 10**(1.0/10.0*snr_db)
    
    # Loop over all the samples
    for idx, row in df_audioset.iterrows():
        
        # Load the sample and add noise 
        audio_data_wav = lb.core.load(row["filename_wav"], sr = sr, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]        
        audio_data_wav_noisy = add_noise(audio_data_wav, snr_pow)
                
        filename_wav_new = wd_noisy + basename(row["filename_wav"])
        
        # We loaded and process the data in float32, but we need to store it in int16        
        audio_data_wav_noisy = np.array(audio_data_wav_noisy / np.max(np.abs(audio_data_wav_noisy)) * np.iinfo(np.int16).max, dtype = "int16")
        lb.output.write_wav(filename_wav_new, audio_data_wav_noisy, sr)
        df_audioset_noisy.loc[idx, "filename_wav"] = filename_wav_new
    
    # Store the data frame with all the info
    df_audioset_noisy.to_pickle(wd_noisy + filename_df_audioset)
    
def main_copy_verified_segments(wd):
    ''' 
    Read the names of the files that have been manually checked and 
    copy them in the subfolder.
    Recreate the info.pkl dataframe.
    '''
    
    wd_verified = wd + "VerifiedDataset\\"

    filename = []
    full_filename = []
    classification = []
    with open(wd_verified + "verified_segments.csv", 'rb') as csvfile:
        spamreader = reader(csvfile)
        for row in spamreader:
                filename.append('sample_{}.wav'.format(row[0]))
                full_filename.append('{}{}'.format(wd_verified, filename[-1]))            
                classification.append(int(row[1]))
 
    df = pd.DataFrame.from_dict({"filename_wav":full_filename, "classification":classification})
    df = df[["filename_wav", "classification"]]
    df = df.assign(valid=np.ones(len(df), dtype=bool))

    df.to_pickle(wd_verified + FILENAME_INFO)

    for f in filename:
        copyfile('{}{}'.format(wd, f), '{}{}'.format(wd_verified, f))        
    
def main_record(wd, wd_recorded, sr):
    '''
    Record all the verified segments.
    ''' 
    
    # Read the dataframe with all the samples info
    filename_df_audioset = FILENAME_INFO
    df_audioset = pd.read_pickle(wd + filename_df_audioset)
    df_audioset_recorded = df_audioset     
        
    # Loop over all the samples
    for idx, row in df_audioset.iterrows():
        
        filename_wav_new = wd_recorded + basename(row["filename_wav"])

        utils.start_and_record(row["filename_wav"], filename_wav_new, sr=sr)
                
        df_audioset_recorded.loc[idx, "filename_wav"] = filename_wav_new
        sleep(1.0)        
    
    # Store the data frame with all the info
    df_audioset_recorded.to_pickle(wd_recorded + filename_df_audioset)
    
def main_calibrate_sound_level(wd):
    '''
    Visualise the "average" sound level and the silence threshold.
    '''
    from pyaudio import PyAudio
    import matplotlib.pyplot as plt
    
    # Start by plot the distribution of the sound level for the recorded pieces 
    df_audioset = pd.read_pickle(wd + FILENAME_INFO)        
    
    hop_size = 4096
    
    avgs, db_avgs, db_maxs = ([] for i in range(3))
    for idx, row in df_audioset.iterrows():            
        audio_data_wav = lb.core.load(row["filename_wav"], sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
        idx = 0        
        while idx+hop_size < len(audio_data_wav):
            audio_data_wav_tmp = audio_data_wav[idx:idx+hop_size]
            audio_data_wav_tmp = np.abs(audio_data_wav_tmp)
            avgs.append(np.average(audio_data_wav_tmp))
            db_avgs.append(20 * np.log10(np.average(audio_data_wav_tmp)))
            db_maxs.append(20 * np.log10(np.max(audio_data_wav_tmp)))
            idx += hop_size
            
    plt.hist(db_avgs)
    plt.hist(db_maxs)
    
    # Now, listen to the mic and print the sound level
    channels = 1
    chunk = hop_size
    audio_format = utils.AUDIO_FORMAT_DEFAULT
         
    audio = PyAudio()
     
    # Start listening
    stream = audio.open(format=utils.AUDIO_FORMAT_MAP[audio_format][1], 
                        channels=channels,
                        rate=utils.SR, 
                        input=True,
                        frames_per_buffer=chunk)
        

    
     
    while True:
        audio_data = stream.read(chunk)
        audio_data = np.fromstring(audio_data, dtype=utils.AUDIO_FORMAT_MAP[audio_format][0])
        audio_data = np.abs(audio_data)
        avg = np.average(audio_data)
        db_avg = 20 * np.log10(np.average(audio_data))
        db_max = 20 * np.log10(np.max(audio_data))
        print('{:.1f}   {:.1f}   {:.1f}'.format(db_avg, db_max, avg))              
 
def parse_audioset(filename_audioset):
    """
    Parse the AudioSet .csv data. We cannot use the pd.read_csv as the quotechar 
    argument does not seem to do its job somehow.
     
    Return a dataframe
    """
     
    yt_id = []
    start_sec = []
    end_sec = []
    positive_labels = []
    with open(filename_audioset, 'rb') as csvfile:
        spamreader = reader(csvfile)#, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
        for row in spamreader:
            if row[0][0] != '#':
                yt_id.append(row[0])
                start_sec.append(float(row[1]))
                end_sec.append(float(row[2]))
                positive_labels.append(','.join(row[3:]).replace('"',''))
     
    df = pd.DataFrame.from_dict({"yt_id":yt_id, "start_sec":start_sec, "end_sec":end_sec, "positive_labels":positive_labels})
    df = df[["yt_id", "start_sec", "end_sec", "positive_labels"]]
     
    # Clean the white space in the labels
    df['positive_labels'] = df['positive_labels'].str.strip()
                 
    return(df)
 
def get_random_dataset(df):
    '''
    Based on the entire dataframe of the parsed AudioSet infos, we build a random sample
    with he desired characteristics.
    We build the sample with replacement.
    '''
     
    def filter_fun(row, target, strict_inclusion):
        """ 
        Apply this filter to find the target labels.
        If strict_inclusion then true if all the positive_labels are in the target 
        If NOT strict_inclusion then true if any of the positive label is in the target
        """ 
        positive_labels = row.positive_labels.split(',')
        if strict_inclusion:            
            idx_valid = all(x in target for x in positive_labels)
        else:
            idx_valid = any(x in target for x in positive_labels)
                 
        return(idx_valid) 
     
    # Fix the seed for now
    np.random.seed(1)  
     
    # Total number of sample and the breakdown of the desired samples
    nb_sample_tot = 5000
 
    dataset_distribution = [
        {"name": "Piano", "id": ['/m/05r5c', '/m/05148p4', '/m/01s0ps', '/m/013y1f'], "strict_inclusion": True, "nb_sample":nb_sample_tot / 2, "classification":1},
        {"name": "Silence,Noise,Speech", "id": ['/m/028v0c', '/m/096m7z', '/m/09x0r'], "strict_inclusion": False, "nb_sample":nb_sample_tot / 2, "classification":0},
    ]              
     
    # Build the output dataframe
    df_all = []
    for item in dataset_distribution:
        # Extract the samples that contained the target label. They may have other extra labels        
        df_tmp = df.loc[df.apply(lambda row: filter_fun(row, item["id"], item["strict_inclusion"]), axis=1),:]
         
        # Add classication for later use
        df_tmp = df_tmp.assign(classification=item["classification"])
         
        # Random sample with replacement
        idx_tgt = np.random.randint(0, len(df_tmp.index), item["nb_sample"])  
        df_tmp = df_tmp.iloc[idx_tgt]
         
        # Add the name of the label
        df_tmp = df_tmp.assign(name=[item["name"]]*len(df_tmp) )
        df_all.append(df_tmp)           
     
    # Return as dataframe and reset the index
    df_all = pd.concat(df_all).reset_index()
     
    return(df_all)
 
def main_build_dataset(wd, filename_audioset, filename_df_random_audioset):
    '''
    Entry-point to:
    1) Parse the AudioSet ".csv" file.
    2) Extract some random samples with the desired properties.
    3) Download these samples from Youtube.
    '''
     
    df_audioset = parse_audioset(filename_audioset)
    df_random_audioset = get_random_dataset(df_audioset)
     
    filename_wav_out_base = wd + "sample_{}.wav"
     
    valids = []
    filenames_wav = []
    for idx, row in df_random_audioset.iterrows():
        filenames_wav.append(filename_wav_out_base.format(idx))
        is_err = utils.youtube_download_audio(row["yt_id"], row["start_sec"], row["end_sec"]-row["start_sec"], filenames_wav[-1])
        valids.append(is_err)
         
    df_random_audioset = df_random_audioset.assign(valid=valids, filename_wav=filenames_wav)
    df_random_audioset.to_pickle(filename_df_random_audioset)
 
    return() 

def audio_data_features_all(df_info, sr, nb_sample, config_features):
    '''
    The function gets the audio data from disk and extract the relevant features 
    to input in the neural network. The nb sample variable correspond to the 
    exact number of samples we extract from the raw audio file (we start 
    from the beginning of the raw waveform). This is required as the NN only 
    understands fixed length features.  
    '''
    classification = []
    features = []
    filename_wav = []
    segment_nb = []
    idx = 0  
    print("Extracting samples features for MLP calibration")  
    for _, row in df_info.iterrows():
        idx += 1
        print("{}/{} done \r".format(idx, len(df_info))) 
        
        if row["valid"]:
            if isfile(row["filename_wav"]):           
                audio_data_wav = lb.core.load(row["filename_wav"], sr = sr, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]

                # Trim the leading silence if we expect music
                if row['classification'] > 0:
                    first_valid = np.where(audio_data_wav > 0.05)[0]
                    audio_data_wav = audio_data_wav[first_valid[0]:len(audio_data_wav)] if len(first_valid) else np.zeros([0]) 
                
                # Look over fixed-size segments
                for k in np.arange(0, len(audio_data_wav)-nb_sample, nb_sample):                    
                    
                    audio_data_wav_tmp = audio_data_wav[k:k+nb_sample]
                    
                    # Don't process if we only have zeros
                    if np.any(audio_data_wav_tmp > 0.0):
                        
                        # Don't include the silence in the music
                        if (not check_is_silence(audio_data_wav_tmp)[0,0]) or int(row['classification']) == 0:
                            (features_one_sample, idxs_normalise, map_config_features) = get_features(audio_data_wav_tmp, 
                                                                                                      nb_sample, 
                                                                                                      config_features)                    
                            classification.append(int(row['classification']))                    
                            features.append(features_one_sample)                    
                            filename_wav.append(row['filename_wav'])
                            segment_nb.append(k)
                    
            else:
                # All the files with row["valid"] = True should be there, 
                # print if this is not he case
                print("File {} is missing".format(row["filename_wav"]))
                
    # Stack the features (samples, time, features)
    classification = np.array(classification)
    classification = classification[:, np.newaxis]    
    features = np.stack(features)
    
    # Normalise the features that need to
    means = np.expand_dims(np.mean(features[:,:,idxs_normalise], axis=0), axis=0)
    stds = np.expand_dims(np.std(features[:,:,idxs_normalise], axis=0), axis=0)
    features[:,:,idxs_normalise] = np.divide(features[:,:,idxs_normalise] - means, stds)
    features_info = {'means':means, 
                     'stds':stds, 
                     'idxs_normalise':idxs_normalise, 
                     'filename_wav':filename_wav, 
                     'segment_nb':segment_nb,
                     'config_features':config_features,
                     'map_config_features':map_config_features}
                 
    return((classification, features, features_info, ))

def get_features(audio_data_wav, nb_sample, config_features):
    
    assert len(audio_data_wav) == nb_sample, 'Length mismatch'
    
    # First, check if we need to get the STFT.    
    config_features_spectrum_based = [x for x in config_features if x["spectrum_based"]]
    any_spectrum_based = len(config_features_spectrum_based) > 0 
    
    # Compute the STFT if we need to
    if any_spectrum_based:
        n_ffts = [x['fcn_kwargs']["n_fft"] for x in config_features_spectrum_based]
        hop_lengths = [x['fcn_kwargs']["hop_length"] for x in config_features_spectrum_based]
        
        # Check if the config has been properly specified
        assert n_ffts[1:] == n_ffts[:-1]
        assert hop_lengths[1:] == hop_lengths[:-1]
        
        # Compute the magnitude spectrogram, it will be reused
        (S,_) = lb.spectrum._spectrogram(y=audio_data_wav, n_fft=n_ffts[0], hop_length=hop_lengths[0], power=1) 
        
        # Append the STFT to the kwargs
        for config in config_features:
            if config["spectrum_based"]:
                config['fcn_kwargs']['S'] = S ** config['spectrum_power']          
    
    # Loop over the configs and actually compute the features
    features_one_sample = []
    idxs_normalise = []  
    audio_data_max = np.max(audio_data_wav)
    map_config_features = []  
    for k, config in enumerate(config_features):
        
        # Audio-level normalisation: normalise to constant loudness
        if config['level_normalise'] and audio_data_max > 0:
            audio_data_wav = audio_data_wav / audio_data_max
            if config["spectrum_based"]:
                config['fcn_kwargs']['S'] /= (audio_data_max ** config['spectrum_power'])
            
        # Compute the features
        features_tmp = config['fcn'](y=audio_data_wav, **config['fcn_kwargs'])
        
        if 'post_process' in config:
            features_tmp = config['post_process'](features_tmp)
        
        # Append the features
        features_one_sample.append(features_tmp)
        
        # Keep the indices of the features we need to normalise
        idxs_normalise.append(np.ones(features_tmp.shape[0], dtype=bool) if config['normalise'] 
                              else np.zeros(features_tmp.shape[0], dtype=bool))
        
        # Keep a config_features <--> idx in features map
        map_config_features.append(np.zeros(features_tmp.shape[0], dtype=np.int) + k)
                                  
    features_one_sample = np.vstack(features_one_sample).T
    idxs_normalise = np.concatenate(idxs_normalise)
    map_config_features = np.concatenate(map_config_features)
    
    return((features_one_sample, idxs_normalise, map_config_features))

def cnn_model(input_shape):    
    """
    Set up the Convolutional Neural Network architecture.
    Only convolve in the time dimension.    
    """
    
    from keras.models import Model
    from keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Flatten
            
    n_conv = 98
    
    X_input = Input(shape=input_shape)
          
    X = Conv1D(n_conv, kernel_size=5, strides=1, padding='same')(X_input)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    
    X = Conv1D(n_conv, kernel_size=5, strides=1, padding='same')(X)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    
    X = Flatten()(X)
    X = Dense(100, activation = "relu")(X)
    X = Dense(1, activation = "sigmoid")(X)
      
    model = Model(inputs = X_input, outputs = X)
      
    return(model)

def train_model(X, y, X_pure_test, y_pure_test):
    """
    Fit the model.
    We have three sets:
    X_train: used for training (incl. validation)
    X_test: used for testing, but the data in X_test is similar to X_train (from the same audio samples)
    X_pure_test: used to get a realistic estimation of real life performance (new audio samples)
    """
    
    from keras.callbacks import ModelCheckpoint
    from sklearn.utils import class_weight
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Ensure no distortion due to more unbalanced class distributions 
    class_weights = class_weight.compute_class_weight('balanced', np.unique(np.squeeze(y_train)), np.squeeze(y_train))
    class_weights = {0:class_weights[0], 1:class_weights[1]}

    # Keep best model
    checkpoint_path = utils.WD + "TempForTrash//" + "weights.hdf5"    
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
    
    # Get the CNN arhitecture
    model = cnn_model(X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    
    model.summary()     
                             
    # Fit (to get better performance on X_pure_test, we may need to only do 2 or 3 epochs...)                                                                     
#     model.fit(X_train, y_train, validation_split=0.2, batch_size=500, epochs=300, class_weight=class_weights, callbacks=[checkpointer])
    model.fit(X_train, y_train, validation_split=0.2, batch_size=500, epochs=3, class_weight=class_weights, callbacks=[checkpointer])
    
    model.load_weights(checkpoint_path)
    
    # Print out-of-sample performance statistics
    print(classification_report(y_test, np.round(model.predict(X_test)), digits=4))
    print(classification_report(y_pure_test, np.round(model.predict(X_pure_test)), digits=4))
    
    return(model)

def main_build_model(wd, sr, nb_sample, config_features):
    '''
    Entry point to build the neural network mode:
    1) Get the audio data from disk and extract the relevant features.
    2) Fit the neural network.
    '''       
    
    df_train = build_df_audio_data("train")
    df_test = build_df_audio_data("test")
    
    # Extract the features and the classifications
    nb_sample = 12288 
    sr = utils.SR
    config_features = get_config_features(utils.SR, 1024, 512, idxs=[0,1,2])
    (y, X, features_info) = audio_data_features_all(df_train, sr, nb_sample, config_features)
    (y_pure_test, X_pure_test, _) = audio_data_features_all(df_test, sr, nb_sample, config_features)
       
    # Fit the neural network model
    model = train_model(X, y, X_pure_test, y_pure_test)    
    
    return(model, features_info)
         
class MusicDetecter():
    def __init__(self, wd, sr, force_rebuild=False):
         
        filename_model = wd + "music_recognition_model.h5"
        filename_model_info = wd + "music_recognition_model_info.pkl"     
        nb_sample = 12288
        self.config_features = get_config_features(utils.SR, 1024, 512) # Set to lower values to get meaningful stddev
         
        # If the model does not exists on disk, re-buid it and store it
        if not isfile(filename_model) or force_rebuild:
                         
            model, features_info = main_build_model(wd, sr, nb_sample, self.config_features)            
            
            features_info = {'means':features_info['means'], 'stds':features_info['stds'],'idxs_normalise': features_info['idxs_normalise']} 
            model_info = {'features_info': features_info, 'sr': sr, 'nb_sample': nb_sample}
            
            # Store to disk
            model.save(filename_model)            
            dump(model_info, open(filename_model_info, 'wb'))   
         
        # Retrieve the model from disk
        self.model = load_model(filename_model)
        self.model._make_predict_function() # (required for multi-threading)
        self.__dict__.update(load(open(filename_model_info, 'rb')))
         
        # Check that the model is as expected
        if self.sr != sr or self.nb_sample != nb_sample:
            raise ValueError("The input parameters do not match with the ones of the model stored on the disk.")
         
        # Set placeholder for the features and the predictions
        self.features = None
        self.predictions = []
        self.audio_data = None
        
    def get_normalised_features(self, audio_data):
        
        features = get_features(audio_data, self.nb_sample, self.config_features)[0] # Only return the features (not the normalisation index)
        features = np.expand_dims(features, axis=0)
        mask = self.features_info['idxs_normalise']
        features[:,:,mask] = np.divide(features[:,:,mask] - self.features_info['means'], self.features_info['stds'])
           
        return(features)
       
    def detect(self, audio_data):
        
        if len(audio_data) < self.nb_sample:
            music_detected = False 
            diagnostic = 'Buffer not yet filled'

        elif check_is_silence(audio_data, threshold=-65, func=lambda x: np.mean(np.abs(x))):
            music_detected = False
            diagnostic = 'Silence'
        
        else:        
            features = self.get_normalised_features(audio_data)            
            music_detected = self.model.predict(features)[0]
            music_detected = True if music_detected > 0.5 else False        
            diagnostic = 'Music' if music_detected else 'Noise/speech' 
            
            # Store the features - REMOVE
            self.predictions.append(music_detected)
            if self.features is None:            
                self.features = features  
                self.audio_data = audio_data         
            else:
                self.features = np.vstack((self.features, features))
                self.audio_data = np.hstack((self.audio_data, audio_data))                
                     
        return(music_detected, diagnostic)
    
if __name__ == '__main__':                       
    music_detecter = MusicDetecter(utils.WD + "Samples\\" , utils.SR)
