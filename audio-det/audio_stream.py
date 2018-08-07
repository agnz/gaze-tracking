import pyaudio
from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import time

#initialize global variables
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
MAX_KNOCK_TIME = 1 #time gap between each knock to be sequential
KNOCK_NUM_TRIGGER = 3 #number of sequential knocks to detect the knock
KNOCK_THRESHOLD = 6900 #thresholding
ALARM_THRESHOLD = 8000

START_TIME = time.time() #get time when program starts

def norm_correlate(stream, template):
	'''
	stream: array of audio samples from real-time
	template: array of audio samples that is to be searched for in stream
	'''
	template = (template - np.mean(template,axis=0))/template.std() #normalize samples
	corr = correlate(stream, template, mode = 'full', method = 'fft')#execute normalized cross correlation
	return corr
	
def sequential_knocks():
	'''
	To be called when a single knock is detected
	Keeps track of the time gap between knocks and then triggers
	'''
	global num_of_knocks
	global last_knock_time
	
	#resets number of knocks if too much time has passed since the last knock
	if time.time()-last_knock_time > MAX_KNOCK_TIME:
		num_of_knocks = 0
	
	num_of_knocks+=1 #increments number of knocks in sequence
	last_knock_time = time.time() #logs down when knock is detected
	print(str(num_of_knocks)) 

	#if number of sequential knocks reaches limit
	if num_of_knocks == KNOCK_NUM_TRIGGER:
		print ("knocks detected at " + str(time.time()-START_TIME) + " seconds")
		num_of_knocks = 0

def alarm_det():
	if time.time()-last_alarm_time > 5:
		print ("alarm detected at "+ str(time.time()-START_TIME) + " seconds")

def callback(in_data, frame_count, time_info, status):
	global knock
	#global alarm
	
	stream = np.fromstring(in_data, dtype = np.int16) #read samples from stream
	stream = stream.reshape(CHUNK,-1) #reshape into numpy array for specified number of channels
	stream = (stream - np.mean(stream,axis=0))/stream.std() #normalize stream samples
	knock_corr = norm_correlate(stream, knock) 
	#alarm_corr = norm_correlate(stream, alarm)
	#print (np.amax(knock_corr))
	if (np.amax(knock_corr))>KNOCK_THRESHOLD:  #thresholding the detection
		sequential_knocks()
	#if (np.amax(alarm_corr))>ALARM_THRESHOLD:
		#alarm_det()
		
	
	return None, pyaudio.paContinue

p = pyaudio.PyAudio()

#read the audio templates
knock = read("knockx3.wav")[1] 
#alarm = read("alarm.wav")[1]

num_of_knocks = 0
last_knock_time = time.time()

#open stream
stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
	stream_callback = callback)

print("* recording")

stream.start_stream()

#never stop the stream
while stream.is_active():
	time.sleep(0.1)


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

