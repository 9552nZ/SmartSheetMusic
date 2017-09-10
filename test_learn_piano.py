import subprocess
import datetime
import pandas as pd
import numpy as np
import csv

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
                
    return(df)

def build_dataset(df):
    '''
    Based on the entire dataframe of the parsed AudioSet infos, we build a random sample
    with he desired characteristics.
    We build the sample with replacement.
    ''' 
    
    # Fix the seed for now
    np.random.seed(1)  
    
    # Total number of sample and the breakdown of the edsired samples
    nb_sample_tot = 300
    dataset_distribution = [
        {"name": "Piano", "id": "/m/05r5c", "nb_sample":nb_sample_tot / 2},
        {"name": "Silence", "id": "/m/028v0c", "nb_sample":nb_sample_tot / 6},
        {"name": "Noise", "id": "/m/096m7z", "nb_sample":nb_sample_tot / 6},
        {"name": "Speech", "id": "/m/09x0r", "nb_sample":nb_sample_tot / 6}
    ]    
    
    # Build the output dataframe
    df_all = []
    for item in dataset_distribution:
        # Extract the samples that contained the target label. They may have other extra labels
        df_tmp = df.loc[lambda df_in: df_in.positive_labels.str.contains(item["id"]), :]
        
        # Random sample with replacement
        idx_tgt = np.random.randint(0, len(df_tmp.index), item["nb_sample"])  
        df_tmp = df_tmp.iloc[idx_tgt]
        
        # Add the name of the label
        df_tmp = df_tmp.assign(name=[item["name"]]*len(df_tmp) )
        df_all.append(df_tmp)           
    
    # Return as dataframe and reset the index
    df_all = pd.concat(df_all).reset_index()
    
    return(df_all)

# def extract_features
                
wd = "C:\\Users\\Alexis\\Business\\SmartSheetMusic\\\AudioSet\\"
filename_wav_out_base = wd + "YoutubeSamples\\sample_{}.wav"
filename_df_random_audioset = wd + "YoutubeSamples\\info.pkl"
# filename_wav_out_base = "C:\\Users\\Alexis\\Business\\SmartSheetMusic\\\AudioSet\\YoutubeSamples\\sample_{}.wav"
start_sec = 1.0
length_sec = 10.0
# filename_audioset = 'C:\Users\Alexis\Business\SmartSheetMusic\AudioSet\unbalanced_train_segments.csv'
filename_audioset = wd + 'unbalanced_train_segments.csv'
# filename_audioset = 'C:\Users\Alexis\Downloads\eval_segments.csv'

df_audioset = parse_audioset(filename_audioset)
df_random_audioset = build_dataset(df_audioset)

errors_download = []
filenames_wav = []
for idx, row in df_random_audioset.iterrows():
    filenames_wav.append(filename_wav_out_base.format(idx))
    is_err = youtube_download_audio(row["yt_id"], row["start_sec"], row["end_sec"]-row["start_sec"], filenames_wav[-1])
    errors_download.append(is_err)
    
df_random_audioset = df_random_audioset.assign(error_download=errors_download, filename_wav=filenames_wav)
df_random_audioset.to_pickle(filename_df_random_audioset)

a = 1