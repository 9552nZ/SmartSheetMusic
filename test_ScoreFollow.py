import ScoreFollow

sf = ScoreFollow.ScoreFollow()

sf.midifile = r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa.mid'
#sf.midifile = r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\progression.mid'

midObj   = sf.load_midi()
wavObj   = sf.midi2wav(midObj)
features = sf.wav2features(wavObj)

sf.start_following(features)
