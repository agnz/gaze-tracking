import pyaudio
from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import time


CHUNK = 1500
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
MAX_KNOCK_TIME = 1
KNOCK_NUM_TRIGGER = 6
THRESHOLD = 32000


def sequential_knocks():
	global num_of_knocks
	global last_knock_time

	if time.time()-last_knock_time > MAX_KNOCK_TIME:
		num_of_knocks = 0

	num_of_knocks+=1
	last_knock_time = time.time()
	print (str(num_of_knocks))

	if num_of_knocks == KNOCK_NUM_TRIGGER:
		print (str(time.time()))
		print ("knocks detected!")
		num_of_knocks = 0


def callback(in_data, frame_count, time_info, status):
	global knock

	data = np.fromstring(in_data, dtype = np.int16)
	data = data.reshape(CHUNK,-1)
	data = data/data.std()
	corr = correlate(data, knock, mode='full', method = 'fft')
	#print (np.amax(np.sqrt(np.mean(corr**2))))
	if (np.amax(np.sqrt(np.mean(corr**2))))>1100:
		sequential_knocks()
	
	return None, pyaudio.paContinue



p = pyaudio.PyAudio()
knock = read("knock.wav")[1]
knock = knock/knock.std()

num_of_knocks = 0
last_knock_time = time.time()

stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
	stream_callback = callback)

print("* recording")

stream.start_stream()

while stream.is_active():
	time.sleep(0.1)


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

