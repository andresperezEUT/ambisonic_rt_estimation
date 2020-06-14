"""
run_experiment_dev_baseline.py
Run the baseline method on the development set,
to estimate the regression coefficients.
"""

import warnings
warnings.simplefilter('always', UserWarning)

import numpy as np
import scipy.signal
import os
import soundfile as sf
import librosa.core
from utils.datasets import get_audio_files_DSD, get_audio_files_librispeech
from utils.blind_rt60_methods import estimate_blind_rt60
from ctf.ctf_methods import compute_t60


# %% GLOBAL SETUP - MUST MATCH ACROSS FILES!

instrument = 'other'

fs = 8000
sh_order = 1
dimM = (sh_order+1)**2

# Number of iterations
I = 10

# IR frequency band
rt60_f = np.asarray([1000])

# Set to your path
IR_folder_path = '/Users/andres.perez/source/ambisonic_rt_estimation/IRs'
result_folder_path = '/Users/andres.perez/source/ambisonic_rt_estimation/results_dev_baseline'
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)

# %% Dataset

audio_type = 'instrument'

# Length and offset
audio_file_length = 20.  # seconds
audio_file_length_samples = int(audio_file_length * fs)
audio_file_offset = 5. # seconds
audio_file_offset_samples = int(audio_file_offset * fs)

af_start = audio_file_offset_samples
af_end = audio_file_offset_samples + audio_file_length_samples


subset = 'Dev'
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



# %% Result placeholders

result_folder_path = os.path.join(result_folder_path, instrument)
# Create folder in case it doesn't exist
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)

rt_method_idx = 1 # rt10
rt60_true = np.empty(I)
rt60_estimated = np.empty(( I, N ))



# %% Analysis loop

# Iterate over IRs
for ir_idx in range(I):
    print('--------------------------------------------')
    print('ITER: ', ir_idx)

    # Get IR
    ir_file_name = str(ir_idx) + '.wav'
    ir_file_path = os.path.join(IR_folder_path, ir_file_name)
    ir, sr = sf.read(ir_file_path)
    assert sr == fs

    # Compute real groundtruth RT60
    rt60_true[ir_idx] = compute_t60(ir[:, 0], fs, rt60_f)[0, rt_method_idx]

    # Iterate over audio files
    for af_idx, af_path in enumerate(audio_files):
        print(af_idx, af_path)
        
        # Build result file name
        result_file_name = str(ir_idx) + '_' + str(af_idx) + '.npy'
        result_file_path = os.path.join(result_folder_path, result_file_name)

        # %% Perform computation only if file does not exist
        if not os.path.exists(result_file_path):

            # Get dry audio signal
            s_t = librosa.core.load(af_path, sr=fs, mono=True)[0][af_start:af_end]
            if instrument == 'speech_all':
                audio_file_length_samples = s_t.size
                print(audio_file_length_samples)
            # Compute reverberant signal by FFT convolution
            y_t = np.zeros((dimM, audio_file_length_samples))
            for ch in range(dimM):
                y_t[ch] = scipy.signal.fftconvolve(s_t, ir[:,ch])[:audio_file_length_samples]  # keep original length


            # %% Baseline
            rt_estimation_method_idx = 0

            # Parameters
            FDR_time_limit = 0.5

            # STFT, only omni channel
            window_size = 1024
            window_overlap = window_size // 4
            nfft = window_size
            _, _, y_tf_omni = scipy.signal.stft(y_t[0], fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

            # Perform estimation from omni channel
            try:
                baseline_rt60 = estimate_blind_rt60(y_tf_omni, fs, window_overlap, FDR_time_limit)
            except ValueError:
                warnings.warn('af_idx ' + str(af_idx) + ': no FDR. Continue')
                continue

            # Store data
            rt60_estimated[ir_idx, af_idx] = baseline_rt60

            # %% Save results

            result_array = np.asarray([rt60_true[ir_idx], baseline_rt60])
            np.save(result_file_path, result_array)
