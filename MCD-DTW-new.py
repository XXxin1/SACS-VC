import numpy as np
import librosa
import os
import math
import pyworld
import pysptk
from tqdm import tqdm
from nnmnkwii.metrics import melcd

SAMPLING_RATE = 22050
FRAME_PERIOD = 5.0
alpha = 0.65  # commonly used at 22050 Hz
fft_size = 512
mcep_size = 34

def load_wav(wav_file, sr):
    """
    Load a wav file with librosa.
    :param wav_file: path to wav file
    :param sr: sampling rate
    :return: audio time series numpy array
    """
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    return wav

def readmgc(filename):
    # all parameters can adjust by yourself :)
    loaded_wav = load_wav(filename, sr=SAMPLING_RATE)

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=SAMPLING_RATE,
                                   frame_period=FRAME_PERIOD, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    # print("mgc of {} is ok!".format(filename))
    return mgc



# define your location of your own test data !
natural_folder = 'GT/'
synth_folder = 'baseline/' 
# you need to make sure all waveform files in these above folders have the same file name !


_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
s = 0.0
 
framesTot = 0





def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


mcd = []
files= sorted(os.listdir(natural_folder))
for wavID in tqdm(files):
    # print("Processing -----------{}".format(wavID))
	
    filename1 = natural_folder + wavID
    mgc1 = readmgc(filename1)
    filename2 = synth_folder + wavID
    mgc2 = readmgc(filename2)
    ref_vec  = mgc1
    synth_vec  = mgc2
    #print(ref_vec.shape)
    ref_frame_no = len(ref_vec)
    min_cost, wp = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T)              
    result = melcd(mgc1[wp[:,0]], mgc2[wp[:,1]] , lengths=None)
    mcd.append(result)
    




print("MCD = : {:f}".format(np.mean(mcd)))


 

