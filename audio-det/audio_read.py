import pyaudio
from scipy.io.wavfile import read
from scipy.signal import correlate
import numpy as np
import time
import sys
import matplotlib


class templates:
    def __init__(self):
        self.samples = read(sys.argv[1])[1]
        self.knock = self.normalize(read('knockx3_trim.wav')[1])
        self.alarm = self.normalize(read('radar_trim.wav')[1])
        self.chunk_size = 1024

    def normalize(self,samples):
        return (samples-np.mean(samples))/np.std(samples)


def main():
    chunks = split_samples(samples, sounds.chunk_size)
    norm_chunks = [sounds.normalize(chunk) for chunk in chunks]
    
    for thresh in np.linspace(0, 15000, 100):
        knock_correlate = correlate(sounds.knock, norm_chunks, thresh)
        alarm_correlate = correlate(sounds.alarm, norm_chunks, thresh)
        knockTPR, knockFPR = get_TPR_FPR(knock_correlate, GT_knock)
        alarmTPR, alarmFPR = get_TPR_FPR(alarm_correlate, GT_alarm)
    
    print (str(knockTPR))
    
def split_samples(samples, chunk_size):
    padded_samples = np.pad(samples, (0, chunk_size - chunk_size%length(samples)), 'constant')
    chunks = [[padded_samples[i:i+chunk_size] for i in range(length(padded_samples)/chunk_size)]
    return chunks

def correlate(template,chunks,thresh):
    scores = [correlate(template, chunk, mode = 'full', method = 'fft') for chunk in chunks]
    hits = [False if np.amax(score) < thresh else True]
    return hits

def get_TPR_FPR(hits, ground_truth_hits):
    positive_idx = np.where(hits==1)
    TPR = np.sum(np.logical_and(hits[positive_idx], ground_truth_hits[positive_idx]))
    FPR = np.sum(np.logical_xor(hits[positive_idx], ground_truth_hits[positive_idx]))

    TPR /= length(hits)
    FPR /= length(hits)
    return TPR, FPR

sounds = track()
main()
