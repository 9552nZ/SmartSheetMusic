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
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Concatenate, concatenate
from keras.layers import LeakyReLU, MaxPooling1D, Flatten, MaxPooling2D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.metrics.classification import confusion_matrix
import datetime
from ntpath import basename, dirname



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
    
# def detector_cnn1(input_shape):
#     
#     n_conv=98
#     
#     X_input = Input(shape=input_shape)
#           
#     X = Conv1D(n_conv, kernel_size=5, strides=1, padding='same')(X_input)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=2, strides=2)(X)
#     
#     X = Conv1D(n_conv, kernel_size=5, strides=1, padding='same')(X)
#     X = BatchNormalization()(X)                                 
#     X = LeakyReLU(alpha=0.1)(X)                                 
#     X = Dropout(0.8)(X)  
#     X = MaxPooling1D(pool_size=2, strides=2)(X)
#     
#     X = Flatten()(X)
#     X = Dense(100, activation = "relu")(X)
#     X = Dense(1, activation = "sigmoid")(X)
#       
#     model = Model(inputs = X_input, outputs = X)
#       
#     return(model)

def detector_cnn2(input_shape):
     
    n_conv=100
    kernel_size = [1, 2, 3, 5, 7, 10, 25]
    
    submodels = []
    X_inputs = []
    for k in kernel_size:
        
        X_inputs.append(Input(shape=input_shape))                     
        X_tmp = Conv1D(n_conv, kernel_size=k, strides=1, padding='same')(X_inputs[-1])
        # X = BatchNormalization()(X) no batch norm? 
        X_tmp = LeakyReLU(alpha=0.1)(X_tmp)                                 
        X_tmp = Dropout(0.1)(X_tmp)  
        X_tmp = GlobalMaxPooling1D()(X_tmp)
        submodels.append(X_tmp)
    
    X = concatenate(submodels)
    X = Dense(1, activation='sigmoid')(X)
            
    model = Model(inputs = X_inputs, outputs = X)
       
    return(model)            


def logistic_regression(input_shape):
    
    X_input = Input(shape=input_shape)        
    X = Flatten()(X_input)    
    X = Dense(10, activation='relu')(X) 
    X = Dense(1, activation='sigmoid')(X)#kernel_regularizer=regularizers.l2(0.0001)
    
    model = Model(inputs=X_input, outputs=X)            
      
    return(model)

## Research robust features and architecture
df_audioset = pd.read_pickle(utils.WD_AUDIOSET + "VerifiedDataset\\VerifiedDatasetRecorded\\" + music_detection.FILENAME_INFO)
df_smd = pd.read_pickle(utils.WD + "Samples\\SaarlandMusicData\\SaarlandMusicDataRecorded\\" + music_detection.FILENAME_INFO)
df_noise = pd.read_pickle(utils.WD + "Samples\\NoiseRecorded\\" + music_detection.FILENAME_INFO) 
df_noise2 = pd.read_pickle(utils.WD + "Samples\\NoiseRecorded\\" + 'info2.pkl')
df_noise3 = pd.read_pickle(utils.WD + "Samples\\Miscellaneous\\" + 'info.pkl')
df_music1 = pd.read_pickle(utils.WD + "Samples\\TestSet\\" + 'info.pkl')

df_train = pd.concat([
    df_audioset.iloc[np.random.randint(0, len(df_audioset), 50)],
    df_smd.iloc[np.random.randint(0, len(df_smd), 4)],
    df_noise.iloc[0:3],
    df_noise2.iloc[0:100]]).reset_index(drop=True)

df_test = pd.concat([df_music1, df_noise2.iloc[500:700], df_noise.iloc[5:]]).reset_index(drop=True)

nb_sample = 12288
# configs = music_detection.get_config_features(utils.SR, 1024, 512, idxs=[k for k in range(14)])
configs = music_detection.get_config_features(utils.SR, 1024, 512, idxs=[0])

(y_train, X_train, features_info_train) = music_detection.audio_data_features_all(df_train, utils.SR, nb_sample, configs)
(y_test, X_test, features_info_train) = music_detection.audio_data_features_all(df_test, utils.SR, nb_sample, configs)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

class_weights = class_weight.compute_class_weight('balanced', np.unique(np.squeeze(y_train)), np.squeeze(y_train))
class_weights = {0:class_weights[0], 1:class_weights[1]}

model = logistic_regression(X_train.shape[1:])
model = detector_cnn2(X_train.shape[1:])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
idxs = np.random.permutation(X_train.shape[0])
# model.fit(X_train[idxs,:,:], y_train[idxs,:], validation_split=0.3, batch_size=1000, nb_epoch=1000, verbose=0, class_weight=class_weights)
model.fit([X_train[idxs,:,:] for k in range(7)], y_train[idxs,:], validation_split=0.3, batch_size=1000, nb_epoch=1000, verbose=0, class_weight=class_weights)
print(confusion_matrix(y_train, np.round(model.predict(X_train))))

utils.figure(); plt.plot(model.predict(np.linspace(0, 1, 200)[:,np.newaxis, np.newaxis]))
#         callbacks = [EarlyStopping(monitor='val_loss', patience=3)]) 

print(classification_report(model.history.validation_data[1], np.round(model.predict(model.history.validation_data[0])), digits=2))
print(confusion_matrix(model.history.validation_data[1], np.round(model.predict(model.history.validation_data[0]))))

print(classification_report(y_test, np.round(model.predict(X_test)), digits=4))
print(confusion_matrix(y_test, np.round(model.predict(X_test))))

for k in np.unique(features_info_train['map_config_features']):
    mask = k == features_info_train['map_config_features']
#     mask = np.logical_or(features_info_train['map_config_features'] == 2, features_info_train['map_config_features'] == 10) 
    model = logistic_regression(X_train[:,:,mask].shape[1:])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    idxs = np.random.permutation(X_train.shape[0])
    model.fit(X_train[idxs,:,:][:,:,mask], y_train[idxs,:], validation_split=0.3, batch_size=200, epochs=2000, verbose=0, class_weight=class_weights,
              callbacks = [EarlyStopping(monitor='val_loss', patience=3)]) 
    
    print("######### CONFIG {}: TRAIN ###########".format(k))
    print(classification_report(model.history.validation_data[1], np.round(model.predict(model.history.validation_data[0])), digits=2))
    print("######### CONFIG {}: TEST ###########".format(k))
    print(classification_report(y_test, np.round(model.predict(X_test[:,:,mask])), digits=2))
    
