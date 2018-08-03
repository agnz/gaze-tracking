from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play
import numpy as np
from python_speech_features import mfcc


def get_nonsilent_samples():
	sound = AudioSegment.from_file("output_audio.aac","aac")
	samples = sound.get_array_of_samples()
	print(len(samples))
	chunks = split_on_silence(sound, min_silence_len = 60, silence_thresh = -30, keep_silence = 1)
	print('done')
	
	for chunk in chunks:
		play(chunk)
	return chunks


get_nonsilent_samples()
