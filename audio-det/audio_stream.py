import pyaudio
from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import time

#initialize global variables
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
MAX_KNOCK_TIME = 4
KNOCK_NUM_TRIGGER = 3 #number of sequential knocks to detect the knock
KNOCK_THRESHOLD = 7000 #thresholding
ALARM_THRESHOLD = 3000

START_TIME = time.time() #get time when program starts

class track:
    def __init__(self):
        self.knock = read("knockx3_trim.wav")[1] 
        self.alarm = read("radar_trim.wav")[1]
        self.num_of_knocks = 0
        self.num_of_alarm_det = 0
        self.first_knock = time.time()
        self.first_alarm = time.time()
    
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
    
    #resets number of knocks if too much time has passed since the first knock
    if time.time()-glob.first_knock > MAX_KNOCK_TIME:
        glob.num_of_knocks = 0
        glob.first_knock = time.time()
    
    glob.num_of_knocks+=1 #increments number of knocks in sequence
    #print(str(num_of_knocks)) 

    #if number of sequential knocks reaches limit
    if glob.num_of_knocks == KNOCK_NUM_TRIGGER:
        print ("knocks detected at " + str(glob.first_knock-START_TIME) + " seconds")

def alarm_det():

    if time.time()-glob.first_alarm > 2:
        glob.num_of_alarm_det = 0
        glob.first_alarm = time.time()
    

    glob.num_of_alarm_det += 1
    #print(str(num_of_alarm_det))


    if glob.num_of_alarm_det == 3:
        print ("alarm detected at "+ str(glob.first_alarm-START_TIME) + " seconds")


def callback(in_data, frame_count, time_info, status):
    global knock
    global alarm
    
    stream = np.fromstring(in_data, dtype = np.int16) #read samples from stream
    stream = stream.reshape(CHUNK,-1) #reshape into numpy array for specified number of channels
    stream = (stream - np.mean(stream,axis=0))/stream.std() #normalize stream samples
    knock_corr = norm_correlate(stream, glob.knock) 
    alarm_corr = norm_correlate(stream, glob.alarm)
      
    if (np.amax(knock_corr))>KNOCK_THRESHOLD:  #thresholding the detection
        sequential_knocks()
    if (np.amax(alarm_corr))>ALARM_THRESHOLD:
        alarm_det()
        #print (np.amax(alarm_corr))
    return None, pyaudio.paContinue

def main():
    p = pyaudio.PyAudio()
    
    print (glob.num_of_knocks)
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

glob = track()
main()

