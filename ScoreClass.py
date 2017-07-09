from music21 import converter
from music21 import midi
from music21 import environment
from music21 import tempo
import os
import StringIO
import numpy as np


class ScoreClass(object):
    def __init__(self, inputFile = '/MusicData/Fur_Elisa.xml'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.file_path = dir_path+inputFile;
        self.score = converter.parse(self.file_path) 
    
    def show(self):
        self.score.show()
    
    def playWin(self):
        self.score.show('midi')
    
    def play(self):
        sp = midi.realtime.StreamPlayer(self.score)
        sp.play()

    def toMidi(self):
        mf = midi.translate.streamToMidiFile(self.score)
        return mf
    
    def exportMidi(self, appendName = '_mod'):
        fp = self.file_path;
        fpDir = fp[0:len(fp) - 4]
        fpOut = fpDir + appendName + ".mid"
        fp = self.score.write('midi', fp=fpOut)
        return fp
    
    def exportXML(self, appendName = '_mod'):
        fp = self.file_path;
        fpDir = fp[0:len(fp) - 4]
        fpOut = fpDir + appendName + ".xml"
        fp = self.score.write('xml', fp=fpOut)
        return fp
        
    def exportWav(self, appendName = '_mod'):
        fpIn = self.exportXML(appendName)
        fpDir = fpIn[0:len(fpIn) - 4]
        fpOut = fpDir + ".wav"
        
        env = environment.UserSettings()
        musescorePath = env['musicxmlPath']
        musescoreRun = '"' + musescorePath + '" ' + fpIn + " -o " + fpOut + " -T 0 "

        fileLikeOpen = StringIO.StringIO()
        os.system(musescoreRun)
        fileLikeOpen.close()
        return fpOut
    
    def pianoRoll(self):
        self.score.plot()
        
    def toText(self):
        self.score.show('text')
                
    def scaleTempo(self, t):
        self.score.augmentOrDiminish(t, inPlace=True)
        
    def shiftPitch(self, semitones):          
        self.score.transpose(semitones, inPlace=True, recurse=True)
                
    def alterNotes(self, freq=0.1):
        sf = self.score.flat
        for f in sf:
            if (f.__class__.__name__ == 'Note' or f.__class__.__name__ == 'Chord'):
                if np.random.rand()<freq:
                    f.transpose(np.sign(np.random.randn()), True)
                    
    def setTempoLocal(self, T=80, modT=0.25, freq = 0.2):
        for m in self.score.recurse(classFilter = 'Measure'):
            if np.random.rand()<freq:
                relChange = 1+modT*np.random.rand()*2
                if np.random.rand()<0.5:
                    relChange=1/relChange
                newT = T*relChange
                # insert metronome mark
                d = tempo.MetronomeMark(referent=1.0, number=newT)
                m.append(d)
                
    