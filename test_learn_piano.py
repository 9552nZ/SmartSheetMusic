import subprocess
import datetime
import pandas as pd
import numpy as np
import csv
import librosa as lb
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# For pandas console priting. May be removed later
pd.set_option('display.expand_frame_repr', False)

def youtube_download_audio(yt_id, start_sec, length_sec, filename_wav_out):
    '''
    Download an audio sample from Youtube and save it as a ".wav".
    More precisely, we generate the "true" Youtube URL using youtube-dl and 
    we then use ffmpeg to download only the relevant portion.  
    Need to specify the beginning of the sample and its length in secs.
    
    An example in cmd for comparison:
    1) "C:/Program Files (x86)/ffmpeg/bin/youtube-dl.exe" -g "https://youtube.com/watch?v=TP9luRtEqjc" --quiet --extract-audio
    2) Get the ouput of the above and replace {} with it:    
    "C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe" -ss 0:00:10 -i "{}" -t 0:01:40 -acodec pcm_s16le -ac 1 -ar 16000 C:\\Users\\Alexis\\Business\\SmartSheetMusic\\Samples\\blopp.wav
    
    Parameters
    ----------
    yt_id : str
        The Youtube id, that is, anything right to "v=". E.g. "TP9luRtEqjc"
        
    start_sec : float > 0
        The starting second for the sample
        
    length_sec : float > 0
        The length we want to download
        
    filename_wav_out : str
        The address to save the ".wav" file
        
    Returns
    -------
    out : bool
        False if the youtube-dl command has failed. True otherwise.
    '''
    
    # Sample rate for the output ".wav" file
    sr = 11025
    
    # Reformat the start_sec and length_sec 
    start_sec = str(datetime.timedelta(seconds=start_sec))
    length_sec = str(datetime.timedelta(seconds=length_sec))
    
    # The target path for the youtube-dl and ffmpeg executables
    ytdl_exe_path = '"C:/Program Files (x86)/ffmpeg/bin/youtube-dl.exe"'
    ffmpeg_exe_path ='"C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"'
    
    # The root URL for Youtube
    yt_url = "https://youtube.com/watch?v={}".format(yt_id)

    # Build the Youtube command
    ytdl_command = "{} -g {} --quiet --extract-audio".format(ytdl_exe_path, yt_url)
    
    # Need to execute the youtube-dl in a try block as the command fails 
    # if the video has been deleted.
    try:
        real_url = subprocess.check_output(ytdl_command)
        real_url = real_url[0:len(real_url)-1] # remove the return character
        
        # The ffmpeg (may need to change the codec here) 
        ffmpeg_command = '{} -ss {} -i "{}" -t {} -acodec pcm_s16le -ac 1 -ar {} {}'.format(ffmpeg_exe_path, start_sec, real_url, length_sec, sr, filename_wav_out)
    
        subprocess.call(ffmpeg_command)
        out = True
        
    except:
        out = False
        
    return(out)

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
        spamreader = csv.reader(csvfile)#, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
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

def main_build_dataset(filename_audioset, filename_df_random_audioset):
    '''
    Entry-point to:
    1) Parse the AudioSet ".csv" file.
    2) Extract some random samples with the desired properties.
    3) Download these samples from Youtube.
    '''
    
    df_audioset = parse_audioset(filename_audioset)
    df_random_audioset = get_random_dataset(df_audioset)
    
    filename_wav_out_base = wd_samples + "sample_{}.wav"
    
    valids = []
    filenames_wav = []
    for idx, row in df_random_audioset.iterrows():
        filenames_wav.append(filename_wav_out_base.format(idx))
        is_err = youtube_download_audio(row["yt_id"], row["start_sec"], row["end_sec"]-row["start_sec"], filenames_wav[-1])
        valids.append(is_err)
        
    df_random_audioset = df_random_audioset.assign(valid=valids, filename_wav=filenames_wav)
    df_random_audioset.to_pickle(filename_df_random_audioset)

    return()

def audio_data_from_disk(wd, df_info, sr, min_samples):
    '''
    Retrieve the data from disk if it exists, otherwise, reshape the ".wav" files, and
    store them to disk.    
    '''
    
    filename_all_data = "{}all_data.pkl".format(wd)
    
    # Retrieve the data from teh disk if it exists
    if os.path.isfile(filename_all_data):
        with open(filename_all_data, 'rb') as handle:
            all_data = pickle.load(handle)
    
    else:            
        valid_idxs = np.full(len(df_info), False, dtype=bool)
        audio_data_wav_all = []
        for idx, row in df_info.iterrows():
            if row["valid"] :
                if os.path.isfile(row["filename_wav"]):
                    audio_data_wav = lb.core.load(row["filename_wav"], sr = sr)[0]
                    
                    # Discard the sample if is is not long enough
                    if len(audio_data_wav) >= min_samples:
                        
                        # Trim the sample to the desired length 
                        audio_data_wav = audio_data_wav[0:min_samples]
                        
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
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return((all_data["classification"], all_data["audio_data"], all_data["df_info"]))    

def extract_features(wd, df_info, n_chroma = 84, framewise_features=False):
     
    sr = 11025
    hop_length = 1024
    features = []

    # We need to get exactly the same sample length.
    # Hence, we either take the first min_secsseconds of the sample 
    # if it it long enough, or we discard the sample entirely. 
    min_secs = 9 
    min_samples = min_secs*sr
    nb_feature = min_samples/hop_length+1 # The expected size of the chroma_cqt function
    
    (classification, audio_data, df_info) = audio_data_from_disk(wd_samples, df_info, sr, min_samples)
    
    for k in range(audio_data.shape[0]):
        # Extract the relevant features
        feature = lb.feature.chroma_cqt(y=audio_data[k,:], sr=sr, hop_length=hop_length, norm=None, n_chroma=n_chroma)
        
        # Make sure that the  features have the same size for each sample
        if feature.shape != (n_chroma, nb_feature): 
            raise ValueError("Chromagram size not valid")
        
        # Append
        features.append(feature)
        
#     if framewise_features:
#         # We transpose the features and duplicate the clasification flags
#         features = map(lambda x: x.T, features)
#         classification = np.repeat(classification, nb_feature)
#     else:
#         # For n features and tk times, we reshape to 1D so as to get: 
#         # [f(1, t1), f(2, t1),..., f(n, t1), f(1, t2), ..., f(n, t2), ...., f(1, tk), ..., f(n, tk)]  
#         features = map(lambda x: np.reshape(x.T, (min_samples/hop_length+1) * n_chroma), features)
        
#     features = np.vstack(features)
        
    return((classification, features, df_info))
    

wd = "C:\\Users\\Alexis\\Business\\SmartSheetMusic\\\AudioSet\\"
# wd_samples = wd + "YoutubeSamples\\Old\\"
wd_samples = wd + "YoutubeSamples\\"
filename_audioset = wd + 'unbalanced_train_segments.csv'
filename_df_random_audioset = wd_samples + "info.pkl"

if __name__ == '__main__':
    wd_samples = wd + "YoutubeSamples\\"
    filename_df_random_audioset = wd_samples + "info.pkl"
    main_build_dataset(filename_audioset, filename_df_random_audioset)
    
df_random_audioset = pd.read_pickle(filename_df_random_audioset)
   
(y_tmp, X_tmp, df_random_audioset) = extract_features(wd_samples, df_random_audioset, n_chroma = 84, framewise_features=True)

X = map(lambda x: x.T, X_tmp)
X = np.vstack(X)
y = np.repeat(y_tmp, X_tmp[0].shape[1])

# (y, X, df_random_audioset) = extract_features(wd_samples, df_random_audioset, n_chroma = 84, framewise_features=True)
# X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.array(df_random_audioset.index))
# X_train, X_test, y_train, y_test = train_test_split(X, y, )
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 10000)  
   
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=2000, solver='lbfgs')
   
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
   
# print(confusion_matrix(y_test,y_pred))
   
print(classification_report(y_test,y_pred))

df_random_audioset.loc[idx_test[np.where(np.logical_and(y_test == 1, y_pred == 0))],:]
    
#################################################################################
    
idx_train, idx_test = train_test_split(np.arange(len(y_tmp)), train_size = 200)

X_train_l = [X_tmp[i] for i in idx_train]
y_train_l = y_tmp[idx_train] 
X_train_l = map(lambda x: x.T, X_train_l)
X_train_l = np.vstack(X_train_l)

y_train_l = np.repeat(y_train_l, X_tmp[0].shape[1])
mlp_l = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=2000)

mlp_l.fit(X_train_l,y_train_l)

idx_train_h, idx_test_h = train_test_split(idx_test)

# y_pred_l = map(lambda x: mlp_l.predict(x), [X_tmp[i].T[0:30, :] for i in idx_test])
y_pred_l = map(lambda x: mlp_l.predict(x.T), X_tmp)
y_pred_l = np.vstack(y_pred_l)


X_train_h = y_pred_l[idx_train_h, :]
# y_train_h = y_tmp[idx_test][idx_train_h]
y_train_h = y_tmp[idx_train_h] 

X_test_h = y_pred_l[idx_test_h, :]
y_test_h = y_tmp[idx_test_h]

mlp_h = MLPClassifier(hidden_layer_sizes=(1),max_iter=2000)
mlp_h.fit(X_train_h,y_train_h)

y_pred_h = mlp_h.predict(X_test_h)

print(classification_report(y_pred_h,y_test_h))

idx_err = np.where(np.logical_and(y_test_h == 1, y_pred_h == 0))

df_random_audioset1 = df_random_audioset.reset_index()
df_random_audioset1.loc[idx_test_h[idx_err],:]

#################################################################################

X = map(lambda x: np.reshape(x.T, X_tmp[0].shape[1] * X_tmp[0].shape[0]), X_tmp)
X = map(lambda x: x[3000:6000], X)
X = np.vstack(X)
y = y_tmp

X_train, X_test, y_train, y_test = train_test_split(X, y)  
   
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=2000)
   
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)   
   
print(classification_report(y_test,y_pred))


  


