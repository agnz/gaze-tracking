from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import sys


'''
Designed to get the ROC curve of the approach of zero normalized cross correlation
'''
class templates:
    def __init__(self):
        #used to initialize global variables
        self.sample_rate, self.samples = read(sys.argv[1]) #read in the wav file sample rate and samples
        self.knock = self.normalize(read('knockx3_trim.wav')[1]) #knock template
        self.alarm = self.normalize(read('radar.wav')[1]) #phone alarm template
        self.chunk_size = 1024 #chunk size to split up the samples

    def normalize(self,samples):
        '''normalizes the samples by subtracting the mean and divides by the standard deviation'''
        return (samples-np.mean(samples))/np.std(samples)


def main():

    chunks = split_samples(sounds.samples, sounds.chunk_size) #splits up the sample array into an array of arrays of size chunk_size
    norm_chunks = [sounds.normalize(chunk) for chunk in chunks] #normalizes all the chunks seperately
    GT_knock, GT_alarm = get_GT_chunks(len(sounds.samples),'0:34') #get the ground truth arrays to compare
    knock_rate = []
    for thresh in np.linspace(0,5000,2): #iterate through the thresholds

        knock_correlate = x_correlate(sounds.knock, norm_chunks, thresh) #correlates template with chunks
        alarm_correlate = x_correlate(sounds.alarm, norm_chunks, thresh)
        knockTPR, knockFPR = get_TPR_FPR(knock_correlate, GT_knock)#gets the TPR and FPR
        alarmTPR, alarmFPR = get_TPR_FPR(alarm_correlate, GT_alarm)
        knock_rate.append((knockTPR,knockFPR))
        print('Loading')
    print (str(knock_rate))
    
def split_samples(samples, chunk_size):
    '''
    :param samples: array of samples
    :param chunk_size: specified size of chunks
    :return: 2-D array of size (n, chunk_size)
    Essentially splits up the array of samples into chunks
    '''
    padded_samples = np.pad(samples, (0, chunk_size - chunk_size % len(samples)), 'constant')
    chunks = [[padded_samples[i:i+chunk_size]] for i in range(int(len(padded_samples)/chunk_size))]
    print (str(np.shape(chunks)))
    return chunks

def x_correlate(template,chunks,thresh):
    '''
    Correlates the template with each one of the chunks and the chunk is True if the threshold is passed.

    :param template:
    :param chunks:
    :param thresh:
    :return: array of True and False representing if the chunk is determined to be a hit
    '''
    #print(np.size(template))
    #print (np.size(chunks))
    scores = [correlate(template, chunk, mode = 'full', method = 'fft') for chunk in chunks]
    hits = [False if np.amax(scores) < thresh else True]
    return hits

def get_TPR_FPR(hits, ground_truth_hits):
    '''
    Compares the hits from the x_correlate and the ground_truth_hits and returns
    True Positive Rate: where both arrays have True at the same index divided by all ground truth hits
    False Positive Rate: where the hits array has a 1 but the ground truth does not, divided by ground truth non-hits

    Truth Table

    True Positive                           False Positive

    Predicted | GT | TP Value               Predicted | GT | FP Value
    ---------------------------             ---------------------------
        1     |  0 |    0                       1     |  0 |    1
    ---------------------------             ---------------------------
        1     |  1 |    1                       1     |  1 |    0

    True Negative                           False Negative

    Predicted | GT | TN Value               Predicted | GT | FN Value
    ---------------------------             ---------------------------
        0     |  0 |    1                       0     |  0 |    0
    ---------------------------             ---------------------------
        0     |  1 |    0                       0     |  1 |    1

    We can see that the TP value matches the GT value and that the FN value matches the ground truth value
    where the predicted values are 1 and 0, respectively. If we just find the indices where the predicted values
    are True and the indices where predicted values are False, then this calculation is trivial

    :param hits: array of True and False
    :param ground_truth_hits: array of True and False
    :return TPR, FPR: float of True positive rate and False positive rate
    '''
    positive_idx = np.where(hits==True) #get indices where predicted hits are true
    TP = np.sum(ground_truth_hits[positive_idx])
    FP = np.sum(np.logical_not(TP))

    negative_idx = np.where(hits==False) #get indices where predicted values are false
    FN = np.sum(ground_truth_hits[negative_idx])
    TN = np.sum(np.logical_not(FN))

    #Recall calculation
    TPR = TP/(TP+FN)

    #1 - Specificity calculation
    FPR = FP/(TN+FP)

    return TPR, FPR

def get_GT_chunks(length_audio, knock_timestamp):
    '''
    Creates the array of the same size as the samples of chunks.
    The element is False if there is supposed to be no hit during that time
    The element is True if there is supposed to be a hit during that time
    :param length_audio: int
    :param knock_timestamp: str
    :return: knock_GT, alarm_GT
    '''
    knock_GT = np.array([False]*length_audio)
    alarm_GT = np.array([False]*length_audio)
    knock_ss = min_sec_to_sec(knock_timestamp)
    chunk_time = sounds.chunk_size/sounds.sample_rate
    chunk_positive = int(knock_ss/chunk_time)
    #print (chunk_positive)
    knock_GT[chunk_positive:chunk_positive+25] = True
    return knock_GT, alarm_GT

def min_sec_to_sec(timestamp):
    '''
    Converts "mm:ss" to ss
    :param timestamp: str
    :return:int
    '''
    m, s = timestamp.split(':')
    ss = int(m) * 60 + int(s)
    return ss

sounds = templates()
main()
