import pandas as pd
import numpy as np
import librosa as lb
import utils_audio_transcript as utils
from csv import reader
from pickle import load, dump, HIGHEST_PROTOCOL
from os.path import isfile 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
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
 
def audio_data_from_disk(wd, df_info, sr, nb_sample):
    '''
    Retrieve the data from disk if it exists, otherwise, reshape the ".wav" files, and
    store them to disk. 
    The nb sample is the exact number of samples we extract from the audio file.   
    '''
     
    filename_all_data = "{}all_data.pkl".format(wd)
     
    # Retrieve the data from the disk if it exists
    if isfile(filename_all_data):
        with open(filename_all_data, 'rb') as handle:
            all_data = load(handle)
     
    else:            
        valid_idxs = np.full(len(df_info), False, dtype=bool)
        audio_data_wav_all = []
        for idx, row in df_info.iterrows():
            if row["valid"] :
                if isfile(row["filename_wav"]):
                    audio_data_wav = lb.core.load(row["filename_wav"], sr = sr)[0]
                     
                    # Discard the sample if is is not long enough
                    if len(audio_data_wav) >= nb_sample:
                         
                        # Trim the sample to the desired length 
                        audio_data_wav = audio_data_wav[0:nb_sample]
                         
                        # Append
                        audio_data_wav_all.append(audio_data_wav)
                        valid_idxs[idx] =True
                else:
                    # All the files with row["valid"] = True should be there, 
                    # print if this is not he case
                    print "File {} is missing".format(row["filename_wav"])
                     
        # Extract valid indices
        df_info = df_info.loc[valid_idxs]
        classification = df_info.classification.as_matrix()
         
        # Stack all the audio into a np array
        audio_data_wav_all = np.vstack(audio_data_wav_all)  
        all_data = {"classification": classification, "audio_data":audio_data_wav_all, "df_info":df_info}                 
         
        # Store to disk
        with open(filename_all_data, 'wb') as handle:
            dump(all_data, handle, protocol=HIGHEST_PROTOCOL)
             
    return((all_data["classification"], all_data["audio_data"], all_data["df_info"]))    
 
def extract_features(wd, df_info, sr, hop_length, n_chroma, nb_sample):
    '''
    The function gets the audio data from disk and extract the relevant features 
    to input in the neural network. The nb sample variable correspond to the 
    exact number of samples we extract from the raw audio file (we start 
    from the beginning of the raw waveform). This is required as the NN only 
    understands fixed length features.  
    '''
     
    # The expected size of the chroma_cqt function
    nb_segment = utils.calc_nb_segment_stft(hop_length, nb_sample)
     
    # Get the audio data from the disk
    (classification, audio_data, df_info) = audio_data_from_disk(wd, df_info, sr, nb_sample)
     
    # Loop over the Youtube samples
    features = []
    for k in range(audio_data.shape[0]):
         
        # Extract the relevant features
        feature = chroma_cqt(audio_data[k,:], sr, hop_length, n_chroma)
         
        # Make sure that the  features have the same size for each sample
        if feature.shape != (n_chroma, nb_segment): 
            raise ValueError("Chromagram size not valid")
         
        # Append
        features.append(feature)
                
    return((classification, features, df_info))
 
def chroma_cqt(audio_data, sr, hop_length, n_chroma):
    '''
    Wrapper for the chroma CQT function. 
    '''
    feature = lb.feature.chroma_cqt(y=audio_data, sr=sr, hop_length=hop_length, norm=2, n_chroma=n_chroma)
#     feature = lb.feature.mfcc(y=audio_data, sr=sr, n_mfcc=84, hop_length=hop_length)
     
    return(feature)
 
def train_model_two_levels(y_tmp, X_tmp):
    '''
    Train a two-level neural network:
    - the lower level classifies one frame of a chromagram as piano / not piano
    - the higher level classifies a group of low level frames as piano / not piano
     
    It seems that a single-level net can do equivalently good though.
    This function is NOT used.
    ''' 
     
    # Idx train is used for the low-level training (we do not need low-level testing)
    idx_train, idx_test = train_test_split(np.arange(len(y_tmp)), train_size = 200)
     
    # Unbundle the features (low-level) 
    X_train_l = [X_tmp[i] for i in idx_train]    
    X_train_l = map(lambda x: x.T, X_train_l)
    X_train_l = np.vstack(X_train_l)
     
    # We need to duplicate the classification to each feature
    y_train_l = y_tmp[idx_train] 
    y_train_l = np.repeat(y_train_l, X_tmp[0].shape[1])
    mlp_l = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=2000)
     
    # Fit the low-level model
    mlp_l.fit(X_train_l,y_train_l)
     
    # Split the remaining data into train and test for the high-level model  
    idx_train_h, idx_test_h = train_test_split(idx_test)
     
    # Get the predicted classification for all the features using the low-level model
    # We could also select a subset of the features here (e.g. mlp_l.predict(x.T[0:50,:]))     
    y_pred_l = map(lambda x: mlp_l.predict(x.T), X_tmp)
    y_pred_l = np.vstack(y_pred_l)
     
    # The features of the high-level model are the output of the lower-level model    
    X_train_h = y_pred_l[idx_train_h, :]    
    y_train_h = y_tmp[idx_train_h]     
    X_test_h = y_pred_l[idx_test_h, :]
    y_test_h = y_tmp[idx_test_h]
     
    # Train the high-level model
    mlp_h = MLPClassifier(hidden_layer_sizes=(100),max_iter=2000)
    mlp_h.fit(X_train_h,y_train_h)
     
    y_pred_h = mlp_h.predict(X_test_h)
     
    # Check the stats (got ~90% out-of-sample precision and recall previously)
    print(classification_report(y_pred_h,y_test_h))
     
     
def train_model(y_tmp, X_tmp):
    '''
    Train the neural network. The features are concatenated into a 1D vector.    
    '''
     
    # Reshape the features
    X = map(lambda x: np.reshape(x.T, X_tmp[0].shape[1] * X_tmp[0].shape[0]), X_tmp)
    X = np.vstack(X)
    y = y_tmp
     
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)  
     
    # Initialise the MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=2000)
     
    # Fit the MLP
    mlp.fit(X_train,y_train)
     
    # Print the stats        
    print(classification_report(y_test,mlp.predict(X_test)))
     
    return(mlp)
     
def main_build_model(wd, sr, hop_length, n_chroma, nb_sample=None):
    '''
    Entry point to build the neural network mode:
    1) Get the audio data from disk and extract the relevant features.
    2) Fit the neural network.
    '''    
     
    # Default to 9 secs of music unless specified otherwise. 
    # The Youtube samples last 10 secs.
    if nb_sample is None:
        nb_sample = 9*sr  
         
    filename_df_random_audioset = wd + "info.pkl"    
    df_random_audioset = pd.read_pickle(filename_df_random_audioset)
     
    # Extract the features and the classifications, using the 
    # Youtube samples stored on the disk 
    (y_tmp, X_tmp, df_random_audioset) = extract_features(wd, df_random_audioset, sr, hop_length, n_chroma, nb_sample)
     
    # Fit the neural network model
    mlp = train_model(y_tmp, X_tmp)
     
    return(mlp)
         
class MusicDetecter():
    def __init__(self, wd, sr, hop_length, n_chroma, nb_sample):
         
        filename_model = wd + "music_recognition_model.pkl"
         
        # If the model does not exists on disk, rebuid it and store it
        if not isfile(filename_model):
                         
            mlp = main_build_model(wd, sr, hop_length, n_chroma, nb_sample=nb_sample)
             
            # Add the parameters on the model built in case these change later. 
            mlp.sr = sr
            mlp.hop_length = hop_length
            mlp.n_chroma = n_chroma 
            mlp.nb_sample = nb_sample
             
            # Store to disk
            dump(mlp, open(filename_model, 'wb'))   
         
        # Retrieve the model from disk
        self.mlp = load(open(filename_model, 'rb'))
         
        # Check that the model is as expected
        if self.mlp.sr != sr or self.mlp.hop_length != hop_length or self.mlp.n_chroma != n_chroma:
            raise ValueError("The input parameters do not match with the ones of the model stored on the disk.")
         
        # Set placeholder for the features
        self.features = []
          
         
    def detect(self, audio_data): #, sr, hop_length, n_chroma
         
        if len(audio_data) < self.mlp.nb_sample:
            return(False)
                     
        features = chroma_cqt(np.array(audio_data), self.mlp.sr, self.mlp.hop_length, self.mlp.n_chroma)
         
        # Store the features
        self.features = features
         
        # Flatten the features to 1D. Be careful, the reshaping should match how it was performed for the NN model.
        features_flat = np.reshape(features.T, features.shape[1] * features.shape[0])
         
        # Lastly, we reshape to get a 2D array (with a unique sample).
        features_flat = features_flat.reshape(1, -1)       
         
#         self.mlp.predict(features)
        print self.mlp.predict(features_flat)[0]
         
        return(True)
     
# music_detecter = MusicDetecter(utils.WD_AUDIOSET, utils.SR, utils.HOP_LENGTH, 84, utils.calc_nb_sample_stft(utils.SR, utils.HOP_LENGTH, 4.0))
 
# audio_data_wav = lb.core.load(utils.WD_AUDIOSET+'sample_28.wav', sr = utils.SR)[0]
# audio_data_wav = audio_data_wav[0:utils.calc_nb_sample_stft(utils.SR, utils.HOP_LENGTH, 4)]
# 
# music_detecter.detect(audio_data_wav)    
 
# Run the below off-line, if we want to recreate the dataset from scratch
# (download the samples from Youtube and store the download info).
# We only need to have the 'unbalanced_train_segments.csv' file already stored to disk     
# if __name__ == '__main__':
#     filename_audioset = utils.WD_AUDIOSET + 'unbalanced_train_segments.csv'
#     filename_df_random_audioset = utils.WD_AUDIOSET + "info.pkl"
#     main_build_dataset(utils.WD_AUDIOSET, filename_audioset, filename_df_random_audioset)    
