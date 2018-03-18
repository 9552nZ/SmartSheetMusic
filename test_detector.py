# Load libraries and model
import os
import time 

import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from keras.models import load_model

cdir = r'C:\Users\Niki\Source\SmartSheetMusic'
os.chdir(cdir)
model = load_model('model.h5')

# Setup up microphone and plots
samplerate = 11025
cliplen = 10
hop_size = 512
Ty = 105
buffer = np.zeros((0,1))
framesinbuffer = samplerate*cliplen

def callback(indata, frames, time, status):  
    global buffer
    buffer = np.append(buffer, indata)
    if buffer.shape[0]>framesinbuffer:
        buffer = buffer[-framesinbuffer:]
        
mic = sd.InputStream(channels=1, callback=callback, samplerate=samplerate)
mic.start()
time.sleep(10) # Wait for buffer to fill in

features = librosa.feature.chroma_stft(y=buffer,sr=samplerate, hop_length=hop_size,n_chroma=96)

plt.ion()
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
im = ax1.imshow(features)
ax1.set_title('Rolling Spectrogram')
ax2 = fig.add_subplot(212)
lines = ax2.plot(np.zeros((Ty,)))
ax2.set_ylim([0,1])
ax2.set_xlim([0,Ty])
ax2.set_title('Probability of Music')
ax2.grid()

def update_plot(*args):
    global buffer
    features = librosa.feature.chroma_stft(y=buffer,sr=samplerate, hop_length=hop_size,n_chroma=96)   
    im.set_array(features)
    pred = model.predict(features[np.newaxis,:,:].swapaxes(1,2))
    lines[0].set_ydata(pred.squeeze())
    return [im,lines]


for i in range(100):
    plt.pause(0.01)
    update_plot()

    fig.canvas.draw()

mic.stop()