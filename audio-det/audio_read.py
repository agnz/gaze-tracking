from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import sys



class templates:
    def __init__(self):
        self.samples = read(sys.argv[1])[1]
        self.knock = self.normalize(read('knockx3_trim.wav')[1])
        self.alarm = self.normalize(read('radar.wav')[1])
        self.chunk_size = 1024
        self.sample_rate = 48000

    def normalize(self,samples):
        return (samples-np.mean(samples))/np.std(samples)


def main():

    chunks = split_samples(sounds.samples, sounds.chunk_size)
    norm_chunks = [sounds.normalize(chunk) for chunk in chunks]
    GT_knock, GT_alarm = get_GT_chunks(len(sounds.samples),'7:30')
    knock_rate = []
    for thresh in np.linspace(0,5000,2):

        knock_correlate = x_correlate(sounds.knock, norm_chunks, thresh)
        alarm_correlate = x_correlate(sounds.alarm, norm_chunks, thresh)
        knockTPR, knockFPR = get_TPR_FPR(knock_correlate, GT_knock)
        alarmTPR, alarmFPR = get_TPR_FPR(alarm_correlate, GT_alarm)
        knock_rate.append((knockTPR,knockFPR))
        print('Loading')
    print (str(knock_rate))
    
def split_samples(samples, chunk_size):
    padded_samples = np.pad(samples, (0, chunk_size - chunk_size % len(samples)), 'constant')
    chunks = [[padded_samples[i:i+chunk_size]] for i in range(int(len(padded_samples)/chunk_size))]
    print (str(np.shape(chunks)))
    return chunks

def x_correlate(template,chunks,thresh):
    #print(np.size(template))
    #print (np.size(chunks))
    scores = [correlate(template, chunk, mode = 'full', method = 'fft') for chunk in chunks]
    hits = [False if np.amax(scores) < thresh else True]
    return hits

def get_TPR_FPR(hits, ground_truth_hits):
    positive_idx = np.where(hits==1)
    TPR = np.sum(np.logical_and(hits[positive_idx], ground_truth_hits[positive_idx]))
    FPR = np.sum(np.logical_xor(hits[positive_idx], ground_truth_hits[positive_idx]))

    TPR /= len(ground_truth_hits[ground_truth_hits==1])
    FPR /= len(ground_truth_hits[ground_truth_hits==0])
    return TPR, FPR

def get_GT_chunks(length_audio, knock_timestamp):
    knock_GT = np.array([False]*length_audio)
    alarm_GT = np.array([False]*length_audio)
    knock_ss = min_sec_to_sec(knock_timestamp)
    chunk_time = sounds.chunk_size/sounds.sample_rate
    chunk_positive = int(knock_ss/chunk_time)
    #print (chunk_positive)
    knock_GT[chunk_positive:chunk_positive+10] = True
    return knock_GT, alarm_GT

def min_sec_to_sec(timestamp):
    m, s = timestamp.split(':')
    ss = int(m) * 60 + int(s)
    return ss

sounds = templates()
main()
