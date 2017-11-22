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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from ntpath import basename
from subprocess import Popen
from time import sleep
from shutil import copyfile

FILENAME_INFO = "info.pkl"

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
    
def main_record(wd, sr):
    '''
    Record all the verified segments.
    ''' 
    # The directory to store the recorded samples
    wd_recorded = wd + 'VerifiedDatasetRecorded\\'
    
    # Read the dataframe with all the samples info
    filename_df_audioset = FILENAME_INFO
    df_audioset = pd.read_pickle(wd + filename_df_audioset)
    df_audioset_recorded = df_audioset 
    
    DETACHED_PROCESS = 0x00000008    
        
    # Loop over all the samples
    for idx, row in df_audioset.iterrows():
        
        filename_wav_new = wd_recorded + basename(row["filename_wav"])
        record_length = utils.get_length_wav(row["filename_wav"])
                                
        cmd = r'C:\\Program Files\\MPC-HC\\mpc-hc64.exe {}'.format(row["filename_wav"])      
        p = Popen(cmd,shell=False,stdin=None,stdout=None,stderr=None,close_fds=True,creationflags=DETACHED_PROCESS)
        
        utils.record(record_length + 0.1, 
                     sr=sr, 
                     audio_format="int16", 
                     save=True, 
                     filename_wav_out=filename_wav_new)            
                
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
     
    # Total number of sample and the breakdown of the edsired samples
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
 
def audio_data_features_all(wd, df_info, sr, nb_sample, config_features):
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
#     print "Extracting samples features for MLP calibration"  
    for _, row in df_info.iterrows():
        idx += 1
        print "{}/{} done \r".format(idx, len(df_info)) 
        
        if row["valid"]:
            if isfile(row["filename_wav"]):           
                audio_data_wav = lb.core.load(row["filename_wav"], sr = sr, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
                
                for k in np.arange(0, len(audio_data_wav)-nb_sample, nb_sample):
                    audio_data_wav_tmp = audio_data_wav[k:k+nb_sample]
                    (features_one_sample, idxs_normalise) = get_features(audio_data_wav_tmp, nb_sample, config_features)
                    is_silence = check_is_silence(audio_data_wav_tmp)[0,0]
                    classification.append(int(row['classification']) if not is_silence else 0)                    
                    features.append(features_one_sample)                    
                    filename_wav.append(row['filename_wav'])
                    segment_nb.append(k)
                    
            else:
                # All the files with row["valid"] = True should be there, 
                # print if this is not he case
                print "File {} is missing".format(row["filename_wav"])

    classification = np.array(classification)
    features = np.vstack(features)
    
    # Normalise the features that need to
    means = np.expand_dims(np.mean(features[:,idxs_normalise], axis=0), axis=0)
    stds = np.expand_dims(np.std(features[:,idxs_normalise], axis=0), axis=0)
    features[:,idxs_normalise] = np.divide(features[:,idxs_normalise] - means, stds)
    features_info = {'means':means, 'stds':stds, 'idxs_normalise':idxs_normalise, 'filename_wav':filename_wav, 'segment_nb':segment_nb}
                
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
    for config in config_features:
        # Compute the features
        features_tmp = config['fcn'](y=audio_data_wav, **config['fcn_kwargs'])
        
        if 'post_process' in config:
            features_tmp = config['post_process'](features_tmp)
        
        # Flatten the features to 1D. Be careful, the reshaping should match how it was performed for the NN model.
        features_flat = np.reshape(features_tmp.T, features_tmp.shape[1] * features_tmp.shape[0])
        
        # Append the features
        features_one_sample.append(features_flat)
        
        # Keep the indices of the features we need to normalise
        idxs_normalise.append(np.ones(len(features_flat), dtype=bool) if config['normalise'] else np.zeros(len(features_flat), dtype=bool))
    
    features_one_sample = np.concatenate(features_one_sample)
    idxs_normalise = np.concatenate(idxs_normalise)
    
    return((features_one_sample, idxs_normalise))
        
def main_build_model(wd, sr, nb_sample, config_features):
    '''
    Entry point to build the neural network mode:
    1) Get the audio data from disk and extract the relevant features.
    2) Fit the neural network.
    '''    
         
    filename_df_random_audioset = wd + FILENAME_INFO    
    df_random_audioset = pd.read_pickle(filename_df_random_audioset)
#     df_random_audioset = df_random_audioset.loc[np.random.randint(0, high=len(df_random_audioset), size=4500)]
    
    # Extract the features and the classifications, using the 
    # Youtube samples stored on the disk 
    (y, X, features_info) = audio_data_features_all(wd, df_random_audioset, sr, nb_sample, config_features)
     
    # Fit the neural network model
    mlp = train_model(y, X)
    
    # Keep the features normalisation info in the MLP
    mlp.features_info = features_info
    
    return(mlp)


def get_config_features(sr, n_fft, hop_length):
    '''
    Get the list of all interesting features that can be used to fit the NN model.
    ''' 
    
    diff = np.diff
    pos_diff = lambda x: np.maximum(np.diff(x),0)
    var = lambda x: np.array([[np.var(x)]])
       
    config_features = [ 
        {'fcn': check_is_silence, 'normalise':False, 'spectrum_based':False, 'fcn_kwargs':{}},                                                                        
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},
        {'fcn': mfcc, 'normalise':True, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 'post_process':pos_diff, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},
        {'fcn': spectral_flux, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},    
                                                                                  
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':12, 'tuning':0.0}},                                       
        {'fcn': lb.feature.spectral_centroid, 'normalise':False, 'spectrum_based':True, 'spectrum_power':1, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
        {'fcn': lb.feature.spectral_bandwidth, 'normalise':False, 'spectrum_based':True, 'spectrum_power':1,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm': True}},
        {'fcn': lb.feature.spectral_rolloff, 'normalise':False, 'spectrum_based':True, 'spectrum_power':1,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
        {'fcn': lb.feature.zero_crossing_rate, 'normalise':False, 'spectrum_based':False,
         'fcn_kwargs':{'frame_length':n_fft, 'hop_length':hop_length}},                       
        {'fcn': lb.feature.chroma_stft, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 'post_process':diff, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length, 'norm':2, 'n_chroma':84, 'tuning':0.0}},                                                                                                                                                               
        {'fcn': mfcc, 'normalise':True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':diff,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
        {'fcn': mfcc, 'normalise':True, 'spectrum_based':True, 'spectrum_power':2, 'post_process':pos_diff,
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},
        {'fcn': lb.feature.zero_crossing_rate, 'normalise':False, 'spectrum_based':False, 'post_process':var,
         'fcn_kwargs':{'frame_length':n_fft, 'hop_length':hop_length}},                                           
        {'fcn': spectral_flux, 'normalise':False, 'spectrum_based':True, 'spectrum_power':2, 'post_process':var, 
         'fcn_kwargs':{'sr': sr, 'n_fft':n_fft, 'hop_length':hop_length}},                                                                                                                                                                                         
        ] 
    
    # Only return the config that works the best
    config_features = config_features[0:5] 
    
    return config_features

def mfcc(y, S, sr, n_fft, hop_length):
    '''
    Emulate the librosa MFCC function to accept the pre-computed power spectrum
    '''
        
    S_new = lb.core.spectrum.power_to_db(lb.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length))    
    
    return lb.feature.mfcc(sr=sr, S=S_new, n_fft=n_fft, hop_length=hop_length)

def check_is_silence(y):
    '''
    Check if the input is below the silence threshold
    '''
    db = 20 * np.log10(np.max(y))
    is_silence = np.expand_dims(np.array([db < -50], dtype=bool), axis=0)
        
    return(is_silence)

def spectral_flux(y, S, sr, n_fft, hop_length):
    '''
    Calculate the specrtal flux as an alternative feature
    '''
    # Normalise the power spectrum with L2 (only if S != 0)     
    divisor = np.linalg.norm(S, ord=2, axis=0)
    if np.all(divisor > 0.0):
        S = np.divide(S, divisor)        
     
    # Compute the flux as the temporal change
    flux = np.linalg.norm(np.diff(S), ord=2, axis=0)
    
    # Return as 2D array
    return np.expand_dims(flux, axis=0)

def train_model(y, X, hidden_layer_sizes=(100)):
    '''
    Train the neural network. The features are concatenated into a 1D vector.    
    '''
     
    # Split into training and test sets
    # Fix the randomiser seed for now
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  
     
    # Initialise the MLP, fix the random seed for now
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,max_iter=2000, random_state=1)
     
    # Fit the MLP
    mlp.fit(X_train,y_train)
     
    # Print the stats        
    print(classification_report(y_test,mlp.predict(X_test)))
     
    return(mlp)
         
class MusicDetecter():
    def __init__(self, wd, sr):
         
        filename_model = wd + "music_recognition_model.pkl"        
        nb_sample = 4096
        self.config_features = get_config_features(utils.SR, 1048, 512) # Set to lower values to get meaningful stddev
         
        # If the model does not exists on disk, re-buid it and store it
        if not isfile(filename_model):
                         
            mlp = main_build_model(wd, sr, nb_sample, self.config_features)
            
            mlp.sr = sr
            mlp.nb_sample = nb_sample
            
            # Store to disk
            dump(mlp, open(filename_model, 'wb'))   
         
        # Retrieve the model from disk
        self.mlp = load(open(filename_model, 'rb'))
         
        # Check that the model is as expected
        if self.mlp.sr != sr or self.mlp.nb_sample != nb_sample:
            raise ValueError("The input parameters do not match with the ones of the model stored on the disk.")
         
        # Set placeholder for the features and the predictions
        self.features = None
        self.predictions = []
        self.audio_data = None
        
    def get_normalised_features(self, audio_data):
        
        features = get_features(audio_data, self.mlp.nb_sample, self.config_features)[0] # Only return the features (not the normalisation index)
        mask = self.mlp.features_info['idxs_normalise']
        features[mask] = np.divide(features[mask] - np.squeeze(self.mlp.features_info['means']), np.squeeze(self.mlp.features_info['stds']))
           
        return(features)
       
    def detect(self, audio_data):
         
        if len(audio_data) < self.mlp.nb_sample:
            return(False)                     
        
        features = self.get_normalised_features(audio_data)
                
        # Lastly, we reshape to get a 2D array (with a unique sample).
        features_flat = features.reshape(1, -1)
                      
        prediction = self.mlp.predict(features_flat)[0]
        
        # Store the features - REMOVE
        self.predictions.append(prediction)
        if self.features is None:            
            self.features = features_flat  
            self.audio_data = audio_data         
        else:
            self.features = np.vstack((self.features, features_flat))
            self.audio_data = np.hstack((self.audio_data, audio_data))
         
        print prediction 
        return(True)

    
if __name__ == '__main__':                       
    music_detecter = MusicDetecter(utils.WD_AUDIOSET + "\VerifiedDataset\\VerifiedDatasetRecorded\\" , utils.SR)
    
#     # REMOVE
#     wd = utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\"
#             
#     configs = get_config_features(utils.SR, 1024, 512)
#     mlp = main_build_model(wd, utils.SR, 4096, configs)
    # for config in configs:
    #     print(config['fcn'].__name__) 
    #     mlp = main_build_model(wd, utils.SR, 4096, [config]) 
        
#     music_detecter = MusicDetecter(wd, utils.SR)
    
    # df_info = pd.read_pickle(wd + FILENAME_INFO)
    # df_info = df_info.assign(valid=np.ones(len(df_info), dtype=bool))
    # (classification, features, features_info) = audio_data_features_all(wd, df_info, utils.SR, music_detecter.mlp.nb_sample, music_detecter.config_features)
    # 
    # classification_pred = music_detecter.mlp.predict(features)
    # 
    # print classification_report(classification,classification_pred)
    # confusion_matrix(classification,classification_pred)
    # 
    # plt.plot(np.abs(classification-classification_pred))
    # 
    # mlp2 = train_model(classification, features, (100,100))#0:400
    # plt.plot(np.abs(classification-mlp2.predict(features)))
     
    # audio_data_wav = lb.core.load(utils.WD_AUDIOSET+'sample_28.wav', sr = utils.SR)[0]
    # audio_data_wav = audio_data_wav[0:music_detecter.mlp.nb_sample]
    #  
    # music_detecter.detect(audio_data_wav)    
     
    # Run the below off-line, if we want to recreate the dataset from scratch
    # (download the samples from Youtube and store the download info).
    # We only need to have the 'unbalanced_train_segments.csv' file already stored to disk     
    # if __name__ == '__main__':
    #     filename_audioset = utils.WD_AUDIOSET + 'unbalanced_train_segments.csv'
    #     filename_df_random_audioset = utils.WD_AUDIOSET + FILENAME_INFO
    #     main_build_dataset(utils.WD_AUDIOSET, filename_audioset, filename_df_random_audioset)    
