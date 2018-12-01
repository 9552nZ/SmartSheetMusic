import music_detection
import pandas as pd
import numpy as np
import os
import librosa as lb
import utils_audio_transcript as utils
from importlib import reload
import matplotlib.pyplot as plt
from pickle import load, dump
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Conv2D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, LeakyReLU, MaxPooling1D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from noise_reduction import NoiseReducer
from sklearn.utils import class_weight
from sklearn.metrics.classification import confusion_matrix
import datetime
from ntpath import basename, dirname

def detector_cnn(input_shape):
    
    n_conv=98
    
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

def detector_cnn_shallow(input_shape):
    
    n_conv=98
    
    X_input = Input(shape=input_shape)
          
    X = Conv1D(n_conv, kernel_size=25, strides=1, padding='same')(X_input)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling1D(pool_size=2)(X)
    
    X = Flatten()(X)
    X = Dense(100, activation = "relu")(X)
    X = Dense(1, activation = "sigmoid")(X)
      
    model = Model(inputs = X_input, outputs = X)
      
    return(model)

def detector_cnn_2d(input_shape):
    
    n_conv = 98
    kernel_size = (5,5)
    strides_conv = (1,1)
    strides_pooling = (2,4)
        
    X_input = Input(shape=input_shape + (1,))    
    
    X = Conv2D(n_conv, kernel_size=kernel_size, strides=strides_conv, padding='same')(X_input)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling2D(pool_size=strides_pooling)(X)
    
    X = Conv2D(n_conv, kernel_size=kernel_size, strides=strides_conv, padding='same')(X)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling2D(pool_size=strides_pooling)(X)
    
    X = Conv2D(n_conv, kernel_size=kernel_size, strides=strides_conv, padding='same')(X)
    X = BatchNormalization()(X)                                 
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dropout(0.8)(X)  
    X = MaxPooling2D(pool_size=strides_pooling)(X)
    
    X = Flatten()(X)
    X = Dense(100, activation = "relu")(X)
    X = Dense(1, activation = "sigmoid")(X)
      
    model = Model(inputs = X_input, outputs = X)
      
    return(model)

# def detector_cnn2(input_shape):
#     
#     n_conv = 10
#     kernel_size = 8
#     stride_conv = 1
#     pool_size = 4
#     stride_max_pool = 2
#     X_input = Input(shape=input_shape)
#           
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X_input)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)
#     
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)
#     
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)
#     
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)  
#     
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)  
#     
#     X = Conv1D(n_conv, kernel_size=kernel_size, strides=stride_conv, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=pool_size, strides=stride_max_pool)(X)   
#         
#     X = Flatten()(X)
#     X = Dense(20, activation = "relu")(X)
#     X = Dense(1, activation = "sigmoid")(X)
#       
#     model = Model(inputs = X_input, outputs = X)
#       
#     return(model)
    
def detector_mlp(input_shape):
    
    X_input = Input(shape=input_shape)        
    X = Flatten()(X_input)    
    X = Dense(100, kernel_regularizer=regularizers.l2(0.001))(X)                                        
    X = LeakyReLU(alpha=0.1)(X)                                 
    X = Dense(1, activation = "sigmoid")(X)
     
    model = Model(inputs = X_input, outputs = X)
         
    return(model)

def clean_audio_data():
    
    for j in range(len(df)):
        filename_wav =  df.loc[j].filename_wav
        audio_data = (lb.core.load(filename_wav, sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0])
        
        audio_data = audio_data[0:(len(audio_data) // 1024)*1024]
        noise_reducer = NoiseReducer() 
        noise_reducer.main(audio_data)

        audio_data_denoised = np.array(noise_reducer.audio_data_denoised)
        audio_data_denoised /= np.max(np.abs(audio_data_denoised))
        n_tot = len(audio_data_denoised)
        hop_length = 8192
        volume = np.zeros(n_tot)
        
        for k in np.arange(0, n_tot, hop_length):
            db = 20 * np.log10(np.max(audio_data_denoised[k:k+hop_length]))
            volume[k:k+hop_length] = db
        
        is_silence = np.array(volume < -37, dtype=float)        
        audio_data_clean = np.array(noise_reducer.audio_data)[~is_silence.astype(bool)]        
        utils.write_wav(dirname(filename_wav) + "\\Clean\\" + basename(filename_wav), audio_data_clean, rate=utils.SR)
        
def truncate_audio_data():
    for j in range(len(df)):
        filename_wav =  df.loc[j].filename_wav
        audio_data = (lb.core.load(filename_wav, sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0])
        utils.write_wav(filename_wav, audio_data[5*utils.SR:len(audio_data)-5*utils.SR], rate=utils.SR)

def plot_model_performance(model):        
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches((12,5))
    ax[0].plot(model.history.history['loss'], label = 'Train')
    ax[0].plot(model.history.history['val_loss'], label = 'Validation')
    ax[0].grid()
    ax[0].set_title('Loss')
    ax[0].legend()
    
    ax[1].plot(model.history.history['acc'], label = 'Train')
    ax[1].plot(model.history.history['val_acc'], label = 'Validation')
    ax[1].grid()
    ax[1].set_title('Accuracy')
    ax[1].legend()   

# import os
# classification = []
# filename_wav = []
# valid = []
# wd_smd = utils.WD + "Samples\\SaarlandMusicData\\"
# for file in os.listdir(wd_smd):
#     if file.endswith(".mp3"):
#         classification.append(1)
#         filename_wav.append(os.path.join(wd_smd, "SaarlandMusicDataRecorded\\", os.path.splitext(file)[0] + ".wav"))
#         valid.append(True)
# 
# classification = []
# filename_wav = []
# valid = []        
# wd_rec = utils.WD + "Samples\\NoiseRecorded\\"
# for file in os.listdir(wd_rec):
#     if file.endswith(".wav"):
#         classification.append(0)
#         filename_wav.append(wd_rec + file)
#         valid.append(True)
# df = pd.DataFrame({'classification':classification, 'filename_wav':filename_wav, 'valid':valid})
# df.to_pickle(wd_rec + music_detection.FILENAME_INFO)                
#         
# wd_smd_recorded = utils.WD + "Samples\\SaarlandMusicData\\SaarlandMusicDataRecorded\\" 
# for file in os.listdir(wd_smd_recorded):
#     if file.endswith(".mp3"):        
#         os.rename(wd_smd_recorded + file, wd_smd_recorded + os.path.splitext(file)[0] + ".wav")
#                 
#                 
#          
# df = pd.DataFrame({'classification':classification, 'filename_wav':filename_wav, 'valid':valid})
# df.to_pickle(wd_smd + "SaarlandMusicDataRecorded\\" + music_detection.FILENAME_INFO)


# df = pd.read_pickle(utils.WD + "Samples\\SaarlandMusicData\\SaarlandMusicDataRecorded\\" + music_detection.FILENAME_INFO)
# # df = df.loc[32:]
# # df = df.reset_index(drop=True)
# # music_detection.main_record(wd_smd, wd_smd + "SaarlandMusicDataRecorded\\", utils.SR)
# 
# filename_wav =  df.loc[0].filename_wav
# audio_data = (lb.core.load(filename_wav, sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0])
# 
# wd = utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\"
# music_detecter = music_detection.MusicDetecter(wd, utils.SR)
# nb_sample = 12288  
# classification = []; diagnostic = []; start_sec = []
# for k in np.arange(nb_sample, len(audio_data), nb_sample):        
#     (classification_tmp, diagnostic_tmp) = music_detecter.detect(audio_data[k-nb_sample:k])
#     classification.append(classification_tmp)
#     diagnostic.append(diagnostic_tmp)
#     start_sec.append((k-nb_sample)/float(utils.SR))
# 
# df_res = pd.DataFrame({'classification':classification, 'diagnostic':diagnostic, 'start_sec':start_sec})
# df_res.to_clipboard() 
#      

wd = utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\"
# configs = [get_config_features(utils.SR, 1024, 512)[0]]
configs = music_detection.get_config_features(utils.SR, 1024, 512, idxs=[0,1,2])
filename_df_audioset = wd + music_detection.FILENAME_INFO    
df_audioset = pd.read_pickle(filename_df_audioset)
df_smd = pd.read_pickle(utils.WD + "Samples\\SaarlandMusicData\\SaarlandMusicDataRecorded\\" + music_detection.FILENAME_INFO)
df_noise = pd.read_pickle(utils.WD + "Samples\\NoiseRecorded\\" + music_detection.FILENAME_INFO) 
df_noise2 = pd.read_pickle(utils.WD + "Samples\\NoiseRecorded\\" + 'info2.pkl')
df_noise3 = pd.read_pickle(utils.WD + "Samples\\Miscellaneous\\" + 'info.pkl')
df_music1 = pd.read_pickle(utils.WD + "Samples\\TestSet\\" + 'info.pkl')
# df_noise4 = pd.read_pickle(utils.WD + "Samples\\NoiseRecorded\\" + music_detection.FILENAME_INFO)
# df_noise4 = df_noise4.iloc[4] 
# df_noise2.to_pickle(utils.WD + "Samples\\NoiseRecorded\\" + 'info2.pkl')
# df_noise2 = pd.read_pickle(utils.WD_AUDIOSET + music_detection.FILENAME_INFO)
# df_noise2 = df_noise2.loc[df_noise2.classification == 0].reset_index()
# df_noise2 = df_noise2[['classification', 'filename_wav', 'valid']]

df_all = pd.concat([df_noise3, df_noise2, df_audioset, df_smd, df_noise]).reset_index(drop=True)
# df_all = pd.concat([df_audioset, df_smd, df_noise]).reset_index(drop=True)
# df_all = df_all.loc[np.random.randint(0, len(df_all), 50)].reset_index(drop=True)
# df_all = df_audioset  


nb_sample = 12288
(y, X, features_info) = music_detection.audio_data_features_all(df_all, utils.SR, nb_sample, configs)
y = y[:,np.newaxis]


# (y_no_rec_noise, X_no_rec_noise, features_info_no_rec_noise) = music_detection.audio_data_features_all(df_noise2, utils.SR, nb_sample, configs)
# y = np.concatenate((y_no_rec_noise, y))
# X = np.concatenate((X_no_rec_noise, X))
# y_all = y; X_all = X; features_info_all = features_info;
# y = y_all; X = X_all; features_info = features_info_all;
# y = y_no_rec_noise; X = X_no_rec_noise; features_info = features_info_no_rec_noise  
# X = np.moveaxis(X, [0,1,2], [2,1,0])
    
# y = y[:,np.newaxis,np.newaxis]
#     y = np.tile(np.array([y]).T, (1, 5))
#     y = y[:,np.newaxis]
#     y = np.tile(np.array([y]).T, (1, 5))[:,:,np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y)
          

# X_train_stacked = np.squeeze(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1)))
# y_train_stacked = np.squeeze(np.reshape(y_train, (y_train.shape[0], y_train.shape[1]*y_train.shape[2], 1)))
# X_test_stacked = np.squeeze(np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2], 1)))
# y_test_stacked = np.squeeze(np.reshape(y_test, (y_test.shape[0], y_test.shape[1]*y_test.shape[2], 1)))
# df_tmp = pd.DataFrame({'filename_wav':[b"C:\Users\Alexis\Business\SmartSheetMusic\Samples\NoiseRecorded\sample_recorded1.wav"], 'classification':[0], 'valid':[True]})
# (y_tmp, X_tmp, features_info_tmp) = music_detection.audio_data_features_all(df_tmp, utils.SR, nb_sample, configs)


# model_scikit = MLPClassifier(hidden_layer_sizes=(100),max_iter=2000, random_state=1)
# model_scikit.fit(X_train_stacked, np.squeeze(y_train))
# print(classification_report(y_test, np.round(model_scikit.predict(X_test_stacked))))
    
# model = detector_mlp(X_train.shape[1:])    
model = detector_cnn(X_train.shape[1:])
# model = detector_cnn_shallow(X_train.shape[1:])
# model = detector_cnn_2d(X_train.shape[1:])
# model = detector_cnn([25, 105])
opt = Adam()#lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
checkpoint_path = filepath=utils.WD + "TempForTrash//" + "weights.hdf5"    
checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

model.summary()     
class_weights = class_weight.compute_class_weight('balanced', np.unique(np.squeeze(y_train)), np.squeeze(y_train))
class_weights = {0:class_weights[0], 1:class_weights[1]}
                                                                                                  
   
model.fit(X_train, y_train, validation_split=0.3, batch_size=500, epochs=500, class_weight=class_weights, callbacks=[checkpointer])#callbacks=[EarlyStopping(patience=5)]

nb_train = 2000
model.fit(X_train[0:nb_train,:,:], y_train[0:nb_train,:], validation_split=0.3, batch_size=500, epochs=100)#callbacks=[EarlyStopping(patience=5)], class_weight=class_weights

# model.fit(X_train[0:nb_train,:,:,np.newaxis], y_train[0:nb_train,:], validation_split=0.3, batch_size=500, epochs=500)#callbacks=[EarlyStopping(patience=5)], class_weight=class_weights
# np.random.seed(1337) 
# model.fit(X_train_stacked, y_train, validation_split=0.3, batch_size=200, epochs=100, callbacks=[EarlyStopping(patience=5)], shuffle=False)
model.load_weights(checkpoint_path)

print(classification_report(y_test, np.round(model.predict(X_test)), digits=4))
print(confusion_matrix(y_test, np.round(model.predict(X_test))))
pd.crosstab(pd.Series(y_test[:,0]), pd.Series(np.round(model.predict(X_test))[:,0]), rownames=['True'], colnames=['Predicted'], margins=True)
plot_model_performance(model)

model_info = {'features_info': features_info, 'sr': utils.SR, 'nb_sample': nb_sample}
# Store to disk
filename_model = wd + "music_recognition_model.h5"
filename_model_info = wd + "music_recognition_model_info.pkl"
model.save(filename_model)
dump(model_info, open(filename_model_info, 'wb'))

model.predict(X)



audio_data_noise = lb.core.load(b"C:\Users\Alexis\Business\SmartSheetMusic\AudioSet//sample_4110.wav", sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
# audio_data_noise2 = lb.core.load(b"C:\Users\Alexis\Business\SmartSheetMusic\Samples\NoiseRecorded//sample_recorded4.wav", sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
audio_data_music = lb.core.load(b"C:\Users\Alexis\Business\SmartSheetMusic\AudioSet\VerifiedDataset\VerifiedDatasetRecorded\\sample_4110.wav", sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]

y_pred = model.predict(X)
mask_noise = y[:,0] == 0
np.where(np.logical_and(mask_noise, np.array(y_pred >0.999)[:,0]))

y_pred[mask_noise][np.argmax(y_pred[mask_noise,0])]

np.where(mask_noise)[0][190]
features_info['filename_wav'][190]

# Show some features
rec_noise = np.array([name in df_noise2['filename_wav'].values for name in features_info['filename_wav']])
idxs_no_rec_noise = np.where(rec_noise)[0] 
idxs_rec_noise = np.where(np.logical_and(np.invert(rec_noise), y[:,0] == 0))[0]
idxs_music = np.where(np.logical_and(np.invert(rec_noise), y[:,0] == 1))[0]
nb_show = 50
# utils.figure()
fig, axes = plt.subplots(3, nb_show)
fig.canvas.manager.window.wm_geometry('1432x880+2366+35')
f_s = 5
f_e = 8
vmax = np.max(X[:,:,f_s:f_e])
vmin = np.min(X[:,:,f_s:f_e])
# vmin = -400
# vmax = 400
for k in range(nb_show):
    axes[0, k].pcolor(X[idxs_no_rec_noise[np.random.randint(0, len(idxs_no_rec_noise))],:,f_s:f_e], vmin=vmin, vmax=vmax)
    axes[1, k].pcolor(X[idxs_rec_noise[np.random.randint(0, len(idxs_rec_noise))],:,f_s:f_e], vmin=vmin, vmax=vmax)
    axes[2, k].pcolor(X[idxs_music[np.random.randint(0, len(idxs_music))],:,f_s:f_e], vmin=vmin, vmax=vmax)

    
# Look at the distributions for the different data sets
m_no_rec_noise = np.mean(X[idxs_no_rec_noise,:,:],axis=(0,1))
m_rec_noise = np.mean(X[idxs_rec_noise,:,:],axis=(0,1))
m_music = np.mean(X[idxs_music,:,:],axis=(0,1))

utils.figure(); plt.plot(m_no_rec_noise); plt.plot(m_rec_noise); plt.plot(m_music)


s_no_rec_noise = np.std(X[idxs_no_rec_noise,:,:],axis=(0,1))
s_rec_noise = np.std(X[idxs_rec_noise,:,:],axis=(0,1))
s_music = np.std(X[idxs_music,:,:],axis=(0,1))
utils.figure(); plt.plot(s_no_rec_noise); plt.plot(s_rec_noise); plt.plot(s_music)

 
# Run some analysis on the samples that could not be learnt
model = load_model(b'C:\Users\Alexis\Business\SmartSheetMusic\AudioSet\VerifiedDataset\VerifiedDatasetRecorded\Old//music_recognition_model.h5')
model = load_model(b'C:\Users\Alexis\Business\SmartSheetMusic\AudioSet\VerifiedDataset\VerifiedDatasetRecorded\music_recognition_model.h5')
y_pred = np.round(model.predict(X)).astype(np.bool)
idxs_bad_noise = np.where(np.bitwise_and(np.invert(y_pred), y.astype(np.bool)))[0]
idxs_bad_music = np.where(np.bitwise_and(y_pred, np.invert(y.astype(np.bool))))[0]

filenames = []; times = [];
for i in idxs_bad_music:
#     audio_data_wav = lb.core.load(features_info["filename_wav"][i], sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
#     first_valid = np.where(audio_data_wav > 0.05)[0][0] # RESTORE IF USE idxs_bad_noise 
    first_valid = 0
    times.append(str(datetime.timedelta(seconds=(first_valid + features_info["segment_nb"][i])/float(utils.SR))))
    filenames.append(features_info["filename_wav"][i])
    
pd.DataFrame(filenames, times).to_clipboard()
pd.DataFrame([features_info["filename_wav"][i] for i in idxs_bad_music]).to_clipboard()
print(confusion_matrix(y, np.round(model.predict(X))))
print(confusion_matrix(y_test, np.round(model.predict(X_test))))

filenames_bad_music = [features_info["filename_wav"][i] for i in idxs_bad_music]
start_secs_bad_music = [features_info["segment_nb"][i] / nb_sample for i in idxs_bad_music]
pd.DataFrame(filenames_bad_music, start_secs_bad_music).to_clipboard()

# Plot the features for dodgy samples
utils.figure()
plt.pcolor(X[idxs_bad_noise[11],:,0:84])
plt.pcolor(X[idxs_bad_music[0],:,0:84])
mask_music = np.squeeze((y==1))
mask_noise = np.squeeze((y==0))
plt.pcolor(X[np.where(mask_music)[0][20],:,0:84])
plt.pcolor(X[np.where(mask_noise)[0][20],:,0:84])

plt.pcolor(X[5000,:,:])
plt.pcolor(X[5000,:,0:84])

plt.plot(X[5000,:,91])
plt.pcolor(np.squeeze(features_info["means"]))
plt.pcolor(np.squeeze(features_info["stds"]))
plt.colorbar()

# np.arange(len(X_test))[np.squeeze(np.round(model.predict(X_test)) != y_test)]
# np.arange(len(X_test))[np.round(model_scikit.predict(X_test_stacked)) != np.squeeze(y_test)]

# m2 = load_model(filename_model)
        
reload(music_detection)
wd = utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\"
music_detecter = music_detection.MusicDetecter(wd, utils.SR)

nb_sample = 12288    
wd_smd_recorded = utils.WD + "Samples\\SaarlandMusicData\\SaarlandMusicDataRecorded\\"
filename_wav = []
classification = []
diagnostic = []
start_sec = []
# df = pd.read_pickle(wd_smd_recorded + music_detection.FILENAME_INFO)
wd_audioset = utils.WD_AUDIOSET
df = pd.read_pickle(wd_audioset + music_detection.FILENAME_INFO)
df = df.loc[df.classification == 0].reset_index()
for i, row in df.iterrows():
    if i in range(500,550) and os.path.isfile(row['filename_wav']):
        print(i)
        audio_data_wav = lb.core.load(row['filename_wav'], sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]    
        for k in np.arange(nb_sample, len(audio_data_wav), nb_sample):        
            (classification_tmp, diagnostic_tmp) = music_detecter.detect(audio_data_wav[k-nb_sample:k])
            filename_wav.append(row['filename_wav'])
            classification.append(classification_tmp)
            diagnostic.append(diagnostic_tmp)
            start_sec.append((k-nb_sample)/float(utils.SR))

df_res = pd.DataFrame({'filename_wav':filename_wav, 'classification':classification, 'diagnostic':diagnostic, 'start_sec':start_sec})
df_res.to_clipboard()

df_res.to_csv(wd_smd_recorded + 'res.csv')

audio_data_wav = lb.core.load(wd + 'sample_2916.wav', sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0]
for k in np.arange(nb_sample, len(audio_data_wav), nb_sample):
    (classification_tmp, diagnostic_tmp) = music_detecter.detect(audio_data_wav[k-nb_sample:k])
    print('{} {}'.format(classification_tmp, diagnostic_tmp))
    
    
utils.record(30.0, sr=utils.SR, audio_format="int16", save=True, 
             filename_wav_out=r"C:\Users\Alexis\Business\SmartSheetMusic\TempForTrash//sample_chopin.wav")

    

# Record some more noise samples
utils.figure();
plt.plot(np.concatenate((X[y==0,10,85], X[y==1,10,85])))


df_audioset = pd.read_pickle(utils.WD_AUDIOSET + music_detection.FILENAME_INFO)
df_audioset = df_audioset.loc[df_audioset.classification == 0].reset_index(drop=True)
# idx = np.where(np.array(['sample_4954.wav' in x for x in df_audioset.filename_wav.values]))[0][0] 
# df_audioset = df_audioset.loc[df_audioset.index.values > idx].reset_index(drop=True)

    
from ntpath import basename
from time import sleep
import os.path

filenames_wav_new = []
for idx, row in df_audioset.iterrows():    
    
    filename_wav_new = utils.WD + "Samples\\NoiseRecorded\\" + basename(row["filename_wav"])
    filenames_wav_new.append(filename_wav_new)
#     if os.path.isfile(row["filename_wav"]): 
#         utils.start_and_record(row["filename_wav"], filename_wav_new, sr=utils.SR)
#             
#         filenames_wav_new.append(filename_wav_new)
#         sleep(1.0)

df_noise2 = pd.DataFrame({'classification':np.zeros(len(filenames_wav_new), dtype=np.int), 
                          'filename_wav':filenames_wav_new, 
                          'valid':np.ones(len(filenames_wav_new), dtype=bool)})        

df_audioset_recorded.to_pickle(wd_recorded + filename_df_audioset)

filenames_wav_new = []
for root, dirs, files in os.walk(utils.WD + "Samples\\Miscellaneous\\"):
    for file in files:        
        filenames_wav_new.append(os.path.join(root, file))

audio_data_all = np.concatenate(audio_data)
lb.output.write_wav(utils.WD + "Samples\\Miscellaneous\\" + "sample_DIRHA.wav", audio_data_all, utils.SR, norm=False)            
            

audio_data_tmp = (lb.core.load(utils.WD + "Samples\\Miscellaneous\\" + "263820__klankbeeld__jannie-88-verjaardag-130509-00-later.ogg", sr = utils.SR, dtype=utils.AUDIO_FORMAT_MAP[utils.AUDIO_FORMAT_DEFAULT][0])[0])
lb.output.write_wav(utils.WD + "Samples\\Miscellaneous\\" + "sample_freesound_263820.wav", audio_data_tmp, utils.SR, norm=False)

df_noise3 = pd.DataFrame({'classification':np.zeros(len(filenames_wav_new), dtype=np.int), 
                          'filename_wav':filenames_wav_new, 
                          'valid':np.ones(len(filenames_wav_new), dtype=bool)})

df_noise3.to_pickle(utils.WD + "Samples\\Miscellaneous\\" + 'info.pkl')

filenames_wav_new = []
for _,row in df_noise2.iterrows():    
    if os.path.isfile(row['filename_wav']):
        filenames_wav_new.append(row['filename_wav'])
        
df_noise3.to_pickle(utils.WD + "Samples\\NoiseRecorded\\" + 'info3.pkl')


utils.youtube_download_audio("Y2m-X3KfAPE", 5, 10, r"C:\Users\Alexis\Business\SmartSheetMusic\TempForTrash//sample_chopin2.wav")
df_noise4 = pd.DataFrame({'classification':np.zeros(1, dtype=np.int), 
                          'filename_wav':[r"C:\Users\Alexis\Business\SmartSheetMusic\TempForTrash//sample_chopin2.wav"], 
                          'valid':np.ones(1, dtype=bool)})

df_music2 = pd.DataFrame({'classification':np.ones(25, dtype=np.int), 
                          'filename_wav':[utils.WD+"Samples\\TestSet\\sample_{}.wav".format(k) for k in range(0,25)], 
                          'valid':np.ones(25, dtype=bool)})
df_music2.to_pickle(utils.WD + "Samples\\TestSet\\" + 'info.pkl')

(y1, X1, features_info1) = music_detection.audio_data_features_all(df_noise4, utils.SR, nb_sample, configs)        
model.predict(X1)

utils.figure()
plt.plot(model.predict(X1))

fig, axes = plt.subplots(1, 2)
fig.canvas.manager.window.wm_geometry('1432x880+2366+35')
f_s = 0
f_e = 84
vmax = np.max(X1[:,:,f_s:f_e])
vmin = np.min(X1[:,:,f_s:f_e])
axes[0].pcolor(X1[12,:,f_s:f_e], vmin=vmin, vmax=vmax)
axes[1].pcolor(X1[345,:,f_s:f_e], vmin=vmin, vmax=vmax)


    