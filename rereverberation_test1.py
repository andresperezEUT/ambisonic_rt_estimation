"""
experiment/rereverberation_test1.py
Retrieve different audios to test rereverberation
"""

import numpy as np
from blind_rt60.datasets import get_audio_files_DSD, get_audio_files_librispeech
from methods import estimate_MAR_sparse_parallel
import os
import soundfile as sf
from ctf.ctf_methods import sid_stft2, compute_t60
import librosa
import warnings
import scipy.signal
import matplotlib.pyplot as plt
import csv
# %% setup

I = 1
IR_folder_path = '/Users/andres.perez/source/dereverberation/experiment/IRs'
fs = 8000
sh_order = 1
dimM = (sh_order+1)**2

rt60_f = np.asarray([1000])

window_type = 'hann'
window_size = 128  # samples
hop = 1 / 2  # in terms of windows
window_overlap = int(window_size * (1 - hop))
nfft = window_size

p = 0.25
i_max = 10
ita = 1e-4
epsilon = 1e-8
L = 20  # number of frames for the IIR filter
tau = int(1 / hop)


plot = True
# %% Dataset

audio_type = 'instrument'

# Length and offset
audio_file_length = 20.  # seconds
audio_file_length_samples = int(audio_file_length * fs)
audio_file_offset = 5. # seconds
audio_file_offset_samples = int(audio_file_offset * fs)

af_start = audio_file_offset_samples
af_end = audio_file_offset_samples + audio_file_length_samples

main_path = '/Volumes/Dinge/datasets'
subset = 'Test'
########################

if audio_type == 'instrument':

    instrument_idx = 1
    instruments = ['bass', 'drums', 'other', 'vocals']
    instrument = instruments[instrument_idx]

    # Get audio files
    # Dataset
    audio_files = get_audio_files_DSD(main_path,
                                      mixtures=False,
                                      dataset_instrument=instrument,
                                      dataset_type=subset)

elif audio_type == 'speech':
    instrument = 'speech'
    audio_files_all = get_audio_files_librispeech(main_path, dataset_type=subset)
    sizes = np.empty(len(audio_files_all))
    # Filter out by length
    for af_idx, af in enumerate(audio_files_all):
        s_t, sr_lib = librosa.core.load(af, sr=None, mono=True)
        sizes[af_idx] = s_t.size / sr_lib
    # mask = np.logical_and(sizes > audio_file_length, sizes < audio_file_length+audio_file_offset)
    mask = sizes > audio_file_length+audio_file_offset
    indices = np.argwhere(mask).flatten()
    audio_files = np.asarray(audio_files_all)[indices]

elif audio_type == 'speech_all':
    instrument = 'speech_all'
    audio_files = np.asarray(get_audio_files_librispeech(main_path, dataset_type=subset))
    # full audio clip
    af_start = 0
    af_end = -1


N = len(audio_files)


# %%


# Iterate over IRs
for ir_idx in range(3,4):
    print('--------------------------------------------')
    print('ITER: ', ir_idx)

    # Get IR
    ir_file_name = str(ir_idx) + '.wav'
    ir_file_path = os.path.join(IR_folder_path, ir_file_name)
    ir, sr = sf.read(ir_file_path)
    assert sr == fs

    # early IR
    early_IR = ir[:window_overlap*tau]

    # Compute real groundtruth RT60
    rt60_true = compute_t60(ir[:, 0], fs, rt60_f)[0, 1] # rt10

    # Iterate over audio files
    for af_idx, af_path in enumerate(audio_files[:1]):
        print(af_idx, af_path)

        # Build result file name
        # result_file_name = str(ir_idx) + '_' + str(af_idx) + '.npy'
        # result_file_path = os.path.join(result_folder_path, result_file_name)

        # dry signal
        s_t = librosa.core.load(af_path, sr=fs, mono=True)[0][af_start:af_end]
        # Ensure there is audio
        if np.allclose(s_t, 0):
            warnings.warn('No audio content')
            continue

        # Compute reverberant signal by FFT convolution
        y_t = np.zeros((dimM, audio_file_length_samples))
        for ch in range(dimM):
            y_t[ch] = scipy.signal.fftconvolve(s_t, ir[:,ch])[:audio_file_length_samples]  # keep original length

        # Early reverberant signal
        early_y_t = np.zeros((dimM, audio_file_length_samples))
        for ch in range(dimM):
            early_y_t[ch] = scipy.signal.fftconvolve(s_t, early_IR[:,ch])[:audio_file_length_samples]  # keep original length


        # %% MAR+SID
        # STFT
        _, _, y_tf = scipy.signal.stft(y_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap,
                                       nfft=nfft)

        # MAR dereverberation
        est_s_tf, _, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
        _, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size,
                                        noverlap=window_overlap, nfft=nfft)
        est_s_t = est_s_t[:, :audio_file_length_samples]

        # Parameters
        filtersize = ir.shape[0]
        winsize = 8 * filtersize
        hopsize = winsize / 16


        # IR ESTIMATION FROM ESTIMATED ANECHOIC SIGNAL
        ir_est_derv = np.zeros((filtersize, dimM))
        for m in range(dimM):
            ir_est_derv[:,m] = sid_stft2(est_s_t[m], y_t[m], winsize, hopsize, filtersize)

        # %% RT60 estimation
        # True computed
        est_derv = compute_t60(ir_est_derv[:,0], fs, rt60_f)[0, 1] # rt10
        rt60_estimated = est_derv

# %% PLOT

print('------------------------')
print('RT60 computed (true):', rt60_true)
print('RT60 estimated:', rt60_estimated)

if plot:
    plt.figure()
    plt.plot(ir)
    plt.title('IR')
    plt.grid()

    plt.figure()
    plt.plot(early_IR)
    plt.title('early IR')
    plt.grid()

    plt.figure()
    plt.plot(s_t)
    plt.title('anechoic signal')
    plt.grid()

    plt.figure()
    plt.plot(y_t.T)
    plt.title('reverberant signal')
    plt.grid()

    plt.figure()
    plt.plot(early_y_t.T)
    plt.title('early reverberant signal')
    plt.grid()

    plt.figure()
    plt.plot(est_s_t.T)
    plt.title('dereberberated signal')
    plt.grid()

    plt.figure()
    plt.plot(ir_est_derv)
    plt.title('estimated IR')
    plt.grid()


# %% WRITE FILES

output_folder_path = '/Volumes/Dinge/audio/dereverberation/' + 'test1'
os.mkdir(output_folder_path)

# audio files
path = os.path.join(output_folder_path, 'ir.wav')
sf.write(path, ir, fs)

path = os.path.join(output_folder_path, 'early_ir.wav')
sf.write(path, early_IR, fs)

path = os.path.join(output_folder_path, 'anechoic.wav')
sf.write(path, s_t, fs)

path = os.path.join(output_folder_path, 'reverberant.wav')
sf.write(path, y_t.T, fs)

path = os.path.join(output_folder_path, 'early_reverberant.wav')
sf.write(path, early_y_t.T, fs)

path = os.path.join(output_folder_path, 'dereverberated.wav')
sf.write(path, est_s_t.T, fs)

path = os.path.join(output_folder_path, 'ir_estimated.wav')
sf.write(path, ir_est_derv, fs)

# text file
path = os.path.join(output_folder_path, 't60.csv')
with open(path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['true (measured)', str(rt60_true)] )
    writer.writerow(['estimated', str(rt60_estimated)] )