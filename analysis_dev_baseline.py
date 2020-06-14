"""
analysis_dev_baseline.py
Obtain fitting parameters for the baseline system, based on the experimental results.

##################### RESULT #####################
drums = (8.242079921128573, -2.193882033832822)
vocals = (10.729872914688878, -3.22347120307927)
bass = (10.359737286288485 -3.277817921881511)
other = (11.848966992443225 -4.081261039251299)
speech = (6.661884937528991 -1.4516773850817029)

"""

import numpy as np
import os
from utils.datasets import get_audio_files_DSD, get_audio_files_librispeech
import matplotlib.pyplot as plt
import scipy.optimize
import librosa.core

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Set to your path
result_folder_path = '/Users/andres.perez/source/ambisonic_rt_estimation/results_dev_baseline'
main_path = '/Volumes/Dinge/datasets'  # Path of the dataset


# %% SETUP

fs = 8000

instrument_idx = 1
instruments = ['bass', 'drums', 'other', 'vocals', 'speech']
instrument = instruments[instrument_idx]
result_folder_path = os.path.join(result_folder_path, instrument)


# Number of iterations
I = 10


# Get audio files
subset = 'Dev'
########################

# Length and offset
audio_file_length = 20.  # seconds
audio_file_length_samples = int(audio_file_length * fs)
audio_file_offset = 5. # seconds
audio_file_offset_samples = int(audio_file_offset * fs)


if instrument != 'speech':
    # Get audio files
    # Dataset
    audio_files = get_audio_files_DSD(main_path,
                                      mixtures=False,
                                      dataset_instrument=instrument,
                                      dataset_type=subset)
else:

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


N = len(audio_files)

# %% Get data

# File name: "ir_idx" _ "af_idx"
# Some of the af_idx are missing. That's because baseline didn't work. We will just skip those files across all IRs.
# Resulting file: np.asarray([rt60_true[ir_idx], baseline_rt60])

result_types = ['rt60_true', 'baseline_rt60']
T = len(result_types)

results = np.empty((I, N, T))
results.fill(np.nan)

for i in range(I):
    for a in range(N):

        # Construct file name
        file_name = str(i) + '_' + str(a) + '.npy'
        file_path = os.path.join(result_folder_path, file_name)

        # Ingest it if it exists
        if os.path.exists(file_path):
            results[i, a] = np.load(file_path)


# %% Statistical analysis


# Sort by increasing true RT60
iii = np.argsort(results[:, 0, 0])

# Mean and std
plt.figure()
plt.title('RT60 Estimation - Mean and std')
plt.grid()
plt.xlabel('IR index')
plt.ylabel('RT60 (s)')
x = np.arange(I)
# True measured RT60
plt.plot(x, results[:, 0, 0][iii], '-o', color=colors[0], markersize=4, label='True')

formats = ['--p']
labels = ['Baseline']

t = 1
mean_values = np.nanmean(results[:, :, t][iii], axis=1)
std_values =  np.nanstd (results[:, :, t][iii], axis=1)
plt.errorbar(x+(t/25), mean_values, yerr=std_values, markersize=4,
             c=colors[t], fmt=formats[t-1], label=labels[t-1])


## Linear regression

def line(x, m, n):
    return m * x + n

p0 = 2, 1 # initial guess
popt, pcov = scipy.optimize.curve_fit(line, mean_values, results[:, 0, 0][iii], p0, sigma=std_values, absolute_sigma=True)
yfit = line(mean_values, *popt)
m, n = popt

plt.plot(x, mean_values*m+n, ':o', markersize=4, c=colors[2], label='mean, linear regression')
plt.legend()

print('INSTRUMENT: ', instrument)
print('--------------------------------------------')
print(' m, n : ', m, n)
print(pcov)
var = np.sum(np.diag(pcov))
std = np.sqrt(var) # joint standard deviation is sqrt of sum of variances https://socratic.org/statistics/random-variables/addition-rules-for-variances
print(std)



# %%
##################### ALL TOGETHER #####################

folder_path = os.path.join('/Users/andres.perez/source/dereverberation/experiment/results_dev_baseline')
instruments = ['bass', 'drums', 'other', 'vocals', 'speech']

C = len(instruments)
r = np.empty((I, N, C))
r.fill(np.nan)

for i in range(I):
    for a in range(N):

        # Construct file name
        file_name = str(i) + '_' + str(a) + '.npy'
        for inst_idx, inst in enumerate(instruments):
            file_path = os.path.join(folder_path, inst, file_name)
            # print(file_path)
            # Ingest it if it exists
            if os.path.exists(file_path):
                r[i, a, inst_idx] = np.load(file_path)[-1]

plt.figure()
plt.title('Baseline - mean dev results')
for inst_idx, inst in enumerate(instruments):
    mean_values = np.nanmean(r[:, :, inst_idx][iii], axis=1)
    std_values = np.nanstd(r[:, :, inst_idx][iii], axis=1)
    # plt.errorbar(np.arange(I), mean_values, yerr=std_values, label=inst)
    plt.errorbar(np.arange(I), mean_values, label=inst)
plt.grid()
plt.legend()



# %%
##################### FIGURE 1 - DRUMS AND SPEECH RESULTS #####################

# File name: "ir_idx" _ "af_idx"
# Some of the af_idx are missing. That's because baseline didn't work. We will just skip those files across all IRs.
# Resulting file: np.asarray([rt60_true[ir_idx], baseline_rt60])

instruments = ['speech', 'drums']
# instruments = ['bass', 'drums', 'other', 'vocals', 'speech']
C = len(instruments)

result_types = ['rt60_true', 'baseline_rt60']
T = len(result_types)

results = np.empty((C, I, N, T))
results.fill(np.nan)

result_folder_path = '/Users/andres.perez/source/dereverberation/experiment/results_dev_baseline'

for c, instrument in enumerate(instruments):
    for i in range(I):
        for a in range(N):

            # Construct file name
            file_name = str(i) + '_' + str(a) + '.npy'
            file_path = os.path.join(result_folder_path, instrument, file_name)

            # Ingest it if it exists
            if os.path.exists(file_path):
                results[c, i, a] = np.load(file_path)

# Sort by increasing true RT60
iii = np.argsort(results[0, :, 0, 0])

# Mean and std
plt.figure()
# plt.title('RT60 Estimation - Baseline method')
plt.grid()
plt.xlabel('IR index')
plt.ylabel('RT60 (s)')
x = np.arange(I)
# True measured RT60
plt.plot(x, results[0, :, 0, 0][iii], '-o', color=colors[0], markersize=4, label='true')
plt.xticks(np.arange(I))

markers = ['d', 's', 'd', 's', 'd', 's']

for c, instrument in enumerate(instruments):
    t = 1
    mean_values = np.nanmean(results[c, :, :, t][iii], axis=1)
    std_values =  np.nanstd (results[c, :, :, t][iii], axis=1)
    plt.errorbar(x+(c/10)-(1/20), mean_values, yerr=std_values, markersize=4,
                 c=colors[c+1], linestyle='--', marker=markers[c],
                 elinewidth = 1, capsize=2, label=r'$\bar{T}_{60}$' + ' ' + instruments[c])


    ## Linear regression

    def line(x, m, n):
        return m * x + n

    p0 = 2, 1 # initial guess
    popt, pcov = scipy.optimize.curve_fit(line, mean_values, results[c, :, 0, 0][iii], p0, sigma=std_values, absolute_sigma=True)
    yfit = line(mean_values, *popt)
    m, n = popt

    plt.plot(x+(c/10)-(1/20), mean_values*m+n, linestyle=':', marker=markers[c],
             markersize=4, c=colors[c+1], label=r'$T_{60}$' + ' ' + instruments[c])


    print('INSTRUMENT: ', instrument)
    print('--------------------------------------------')
    print(' m, n : ', m, n)
    print(pcov)
    var = np.sum(np.diag(pcov))
    std = np.sqrt( var)  # joint standard deviation is sqrt of sum of variances https://socratic.org/statistics/random-variables/addition-rules-for-variances
    print(std)
plt.legend()
