# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:22:58 2017

@author: Niki
"""

import pretty_midi
progression = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)
pos = 0;
for note_name in ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6', 'B5', 'A5', 'G5', 'F5', 'E5', 'D5', 'C5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=pos, end=pos+0.5)
    pos+=0.5;
    piano.notes.append(note)
progression.instruments.append(piano)
progression.write('progression.mid')