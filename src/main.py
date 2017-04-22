import numpy as np
import sys
import os
import scipy.io.wavfile as scw
import scipy
import tensorflow as tf

WIN_LENGTH = 2048
OVERLAP = 4

def load_files(noisy, clean):
    for f in os.listdir(noisy):
        yield(make_inputs(load_audio(noisy+f), load_audio(clean+f)))

def make_inputs(noisy, clean):
    n_fft = scipy.absolute(stft(noisy))
    c_fft = scipy.absolute(stft(clean))
    return np.array([[n_fft[i], c_fft[i]] for i in range(len(n_fft))])

def stft(x):
    """Returns short time fourier transform of a signal x
    """
    hop = int(WIN_LEN / OVERLAP)
    w = scipy.hanning(WIN_LEN+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+WIN_LEN]) for i in range(0,len(x)-WIN_LEN, hop)])

def load_audio(path):
    raw = scw.read(path)[1]
    if raw.ndim > 1:
        return raw
    else:
        return raw[:,0]

if __name__ == "__main__":
    noisy_p, clean_p = sys.argv[1:]
    train_set = load_files(noisy_p, clean_p)
