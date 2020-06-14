"""
analysis.py
Evaluate experiment results, and plot figures
"""

import numpy as np
import os
from utils.datasets import get_audio_files_DSD, get_audio_files_librispeech
import matplotlib.pyplot as plt
import librosa.core

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Set to your path
result_folder_path = '/Users/andres.perez/source/ambisonic_rt_estimation/results'
main_path = '/Volumes/Dinge/datasets'  # Path of the dataset


# %% SETUP

fs = 8000

instrument_idx = 1
instruments = ['bass', 'drums', 'other', 'vocals', 'speech']
instrument = instruments[instrument_idx]
result_folder_path = os.path.join(result_folder_path, instrument)

# Number of iterations
I = 9 #excude first IT


# Get audio files
subset = 'Test'
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
# File content: ([rt60_true, baseline_rt60, est_true, est_derv])

result_types = ['rt60_true', 'baseline_rt60', 'est_true', 'est_derv']
T = len(result_types)

results = np.empty((I, N, T))
results.fill(np.nan)

for i in range(I):
    for a in range(N):

        # Construct file name
        file_name = str(i+1) + '_' + str(a) + '.npy'
        file_path = os.path.join(result_folder_path, file_name)
        # Ingest it if it exists
        if os.path.exists(file_path):
            # print(np.load(file_path))
            results[i, a] = np.load(file_path)


################## CORRECT BASELINE DATA ##################

# Those hardcdded values are obtained from `analysis_dev_baseline.py`

corrections = {
    'drums':    (8.242079921128573, -2.193882033832822),
    'vocals':   (10.729872914688878, -3.22347120307927),
    'bass' :    (10.359737286288485, -3.277817921881511),
    'other':    (11.848966992443225, -4.081261039251299),
    'speech':   (6.661884937528991, -1.4516773850817029),
}

# Let's re-scale them
baseline_idx = 1
m, n = corrections[instrument]
results[:,:,baseline_idx] = results[:,:,baseline_idx]*m + n


### adjust number of samples
# in case we didn't finish computation
# finished_irs = np.argwhere(~np.isnan(results[:, -1, -1])).squeeze()
# results = results[finished_irs]
# I = finished_irs.size


################## REMOVE NANS ##################
# There are some missing files, represented in the results array
# as a whole row of nans. This is because there is no signal for this audio file.
# Let's remove them, using the first file
ir_index = 0 # whatever
method_index = 0
not_nan_indices= np.argwhere(~np.isnan(results[ir_index,:,method_index])).squeeze()
results = results[:, not_nan_indices, :]

# %% Check data individually

ir_idx = 0

## Histogram
formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']
kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=np.arange(0,5,0.1), ec="k")
plt.figure()
plt.title('IR '+str(ir_idx))
plt.grid()
for t in range(1,4):
    plt.hist(results[ir_idx, :, t], **kwargs, color=colors[t], label=labels[t-1])
plt.legend()

## Boxplot
plt.figure()
plt.title('IR '+str(ir_idx))
plt.grid()
v = results[ir_idx, :, :]
plt.boxplot(v)




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
plt.plot(x, results[:I, 0, 0][iii], '-o', color=colors[0], label='True')

formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']

for t in range(1,4):
    plt.errorbar(x+(t/25),  np.nanmean(results[:I, :, t][iii], axis=1),
                 yerr=np.nanstd(results[:I, :, t][iii], axis=1),
                 c=colors[t], fmt=formats[t-1], label=labels[t-1])
plt.legend()

# Median
plt.figure()
plt.title('RT60 Estimation - Median')
plt.grid()
plt.xlabel('IR index')
plt.ylabel('RT60 (s)')
x = np.arange(I)
# True measured RT60
plt.plot(x, results[:I, 0, 0][iii], '-o', color=colors[0], label='True')

formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']

for t in range(1,4):
    plt.errorbar(x+(t/25),  np.nanmedian(results[:I, :, t][iii], axis=1),
                 c=colors[t], fmt=formats[t-1], label=labels[t-1])
plt.legend()

# together
plt.figure()
plt.subplot(1,2,1)
plt.grid()
plt.xlabel('IR index')
plt.ylabel('RT60 (s)')
x = np.arange(I)
# True measured RT60
plt.plot(x, results[:I, 0, 0][iii], '-o', color=colors[0], label='True')
formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']
for t in range(1,4):
    plt.errorbar(x+(t/25),  np.nanmean(results[:I, :, t][iii], axis=1),
                 yerr=np.nanstd(results[:I, :, t][iii], axis=1),
                 elinewidth=1, capsize=2,
                 c=colors[t], fmt=formats[t-1], label=labels[t-1])
plt.subplot(1,2,2)
plt.grid()
plt.xlabel('IR index')
# plt.ylabel('RT60 (s)')
x = np.arange(I)
# True measured RT60
plt.plot(x, results[:I, 0, 0][iii], '-o', color=colors[0], label='True')
for t in range(1,4):
    plt.errorbar(x+(t/25),  np.nanmedian(results[:I, :, t][iii], axis=1),
                 c=colors[t], fmt=formats[t-1], label=labels[t-1])
plt.legend()




# %% Histogram - all data

# Exclude any data with nan values!
r_diff = results[:,:,1:] - results[:,:,0,np.newaxis]

## Histogram
formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']
kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=np.arange(-0.525,3,0.05), ec="k")
plt.figure()
plt.title('diff histogram')
plt.grid()
for t in range(3):
    plt.hist(r_diff[:,:,t].flatten(), **kwargs, color=colors[t], label=labels[t])
plt.legend()
plt.yscale('log')


#
## Boxplot
# reshape data
a,b,c = r_diff.shape
plt.figure()
plt.title('diff boxplot')
plt.grid()
plt.boxplot(r_diff.reshape(a*b,c),
            notch=False,
            sym='',
            vert=True,
            labels=labels,
            meanline=True,
            showmeans=True
            )


# %% Check outliers

# labels = ['Baseline', 'Oracle SID', 'MAR + SID']
# plt.figure()
# # plt.suptitle(instrument)
# for i in range(1,4):
#     plt.subplot(3,1,i)
#     # plt.title(labels[i-1])
#     plt.plot(np.arange(1,N+1),results[:, :, 0].T, ':')
#     plt.plot(np.arange(1,N+1),results[:,:,i].T)
#     plt.grid()
# plt.legend()
#
# # Version with the nans
# r = np.empty((I, N, T))
# r.fill(np.nan)
# for i in range(I):
#     for a in range(N):
#         # Construct file name
#         file_name = str(i+1) + '_' + str(a) + '.npy'
#         file_path = os.path.join(result_folder_path, file_name)
#         # Ingest it if it exists
#         if os.path.exists(file_path):
#             r[i, a] = np.load(file_path)
# # Let's re-scale them
# baseline_idx = 1
# m, n = corrections[instrument]
# r[:,:,baseline_idx] = r[:,:,baseline_idx]*m + n
# labels = ['Baseline', 'Oracle SID', 'MAR + SID']
#
#
# plt.figure()
# # plt.suptitle(instrument)
# for q in range(3):
#     plt.subplot(3,1,q+1)
#     print(q)
#     plt.ylabel(labels[q])
#     for i in range(I):
#         plt.plot(np.arange(1,N+1),r[i, :, 0], ':', color=colors[i])
#         plt.plot(np.arange(1,N+1),r[i,:,q+1], '-o', markersize=2, color=colors[i])
#     plt.grid()
#     plt.xticks(np.arange(1,N+1))
# plt.xlabel('Audio clip number')
#
#


# %% ################### boxplot ###################

N = [30,46]
instruments = ['speech', 'drums']

methods = ['Baseline', 'Oracle SID', 'MAR+SID']

C = len(instruments)
r = np.empty((C, I, max(N), 4))
r.fill(np.nan)

r_drums = np.load('/Users/andres.perez/source/dereverberation/experiment/results/r_drums.npy')
r_speech = np.load('/Users/andres.perez/source/dereverberation/experiment/results/r_speech.npy')

r[0, :,:N[0]] = r_speech
r[1, :,:N[1]] = r_drums


# Sort by increasing true RT60
iii = np.argsort(r[0, :, 0, 0])

# Median
plt.figure()

formats = ['--p', ':d', '-.s']
labels = ['Baseline', 'Oracle SID', 'MAR + SID']

for inst_idx, instrument in enumerate(instruments):
    plt.subplot(2,1,inst_idx+1)
    plt.grid()
    plt.plot(x, r[0, :I, 0, 0][iii], '-o', color=colors[0], label='true')
    plt.xticks(np.arange(I))
    plt.ylabel('RT60 (s)')
    x = np.arange(I)
    # True measured RT60

    for t in range(1,4):
        plt.errorbar(x,  np.nanmedian(r[inst_idx, :, :, t][iii], axis=1), markersize=4,
                     c=colors[t], fmt=formats[t-1], label=labels[t-1])
    if inst_idx==0:
        plt.legend()
plt.xlabel('IR index')



# %% ################### boxplot ###################

N = [30,46]
instruments = ['speech', 'drums']


methods = ['Baseline', 'Oracle SID', 'MAR+SID']

C = len(instruments)
r = np.empty((C, I, max(N), 3))
r.fill(np.nan)

v_drums = r_drums[:,:,1:] - r_drums[:,:,0,np.newaxis]
v_speech = r_speech[:,:,1:] - r_speech[:,:,0,np.newaxis]
# v_drums = np.load('/Users/andres.perez/source/dereverberation/experiment/results/r_diff_drums.npy')
# v_speech = np.load('/Users/andres.perez/source/dereverberation/experiment/results/r_diff_speech.npy')

r[0, :,:N[0]] = v_speech[:I]
r[1, :,:N[1]] = v_drums[:I]

th = 1.5

data = []
for method_idx, method in enumerate(methods):
    for inst_idx, inst in enumerate(instruments):
        rrr =  r[inst_idx, :, :, method_idx].flatten()
        # exclude nans
        rrr = rrr[~np.isnan(rrr)]
        # exlude bigger than threshold
        rrr = rrr[np.abs(rrr)<th]
        data.append(rrr[~np.isnan(rrr)])
data = np.asarray(data)


import seaborn as sns

# %% BIG BOXPLOT

import matplotlib.ticker as ticker




# plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
# plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

# fig, ax = plt.subplots(3,2, sharex='col', figsize=plt.figaspect(1.5))
fig, ax = plt.subplots(3,2, sharex='col', figsize=(9,6))
plt.subplots_adjust(hspace = 0, wspace = 0.15)



for t in range(1, 4):

    plt.grid()
    sns.boxplot(data=v_speech[:, :, t - 1], orient='h', fliersize=2, boxprops=dict(alpha=.75), linewidth=1, ax=ax[t-1, 0])
    # sns.boxplot(data=ll, orient='h', fliersize=2, boxprops=dict(alpha=.75), linewidth=1, ax=ax[t-1, 0])
    ax[t-1,0].set_ylabel(methods[t - 1])
    ax[t-1,0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[t-1,0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[t - 1, 0].yaxis.tick_right()
ax[2,0].set_xlim(-0.5, 0.3)
    # ax.xaxis.set_tick_params(length=0)
# ax.xaxis.set_tick_params(length=3)
# plt.xticks(np.arange(-0.5, 0.3, 0.1))
ax[2,0].set_xlabel('RT60 error (s)')


for t in range(1, 4):
    # remove elements bigger than threshold
    ll = [[] for i in range(30)]
    for i in range(30):
        v = v_drums[:, i, t - 1]
        ll[i] = v[np.abs(v)<th]

    # v = v_drums[:, :, t - 1]
    # v = v[np.abs(v)<th]
    sns.boxplot(data= ll, orient='h', fliersize=2, boxprops=dict(alpha=.75), linewidth=1, ax=ax[t-1, 1])
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # plt.xlabel('Audio clip index')

    # ax[t - 1, 1].set_ylabel('Audio clip index')
    ax[t-1,1].yaxis.set_major_locator(ticker.MultipleLocator(8))
    ax[t-1,1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[t - 1, 1].yaxis.tick_right()
ax[2,1].set_xlim(-1, 1.5)
# plt.grid()
plt.xlabel('RT60 error (s)')

ax[0,0].grid(axis='x')
ax[1,0].grid(axis='x')
ax[2,0].grid(axis='x')
ax[0,1].grid(axis='x')
ax[1,1].grid(axis='x')
# ax[2,1].grid(axis='x')

ax[0, 1].yaxis.set_label_position("right")
ax[0, 1].set_ylabel('Audio clip index')
ax[1, 1].yaxis.set_label_position("right")
ax[1, 1].set_ylabel('Audio clip index')
ax[2, 1].yaxis.set_label_position("right")
ax[2, 1].set_ylabel('Audio clip index')





# %% BIG BOXPLOT


# fig, ax = plt.subplots(2,2, sharex='col', figsize=plt.figaspect(1/3))
fig, ax = plt.subplots(2,2, sharex='col', figsize=(9,4))
plt.subplots_adjust(hspace = 0, wspace = 0.15)
# plt.subplot(2,2,1)
# plt.grid()
sns.boxplot(data=data[0::4], orient='h', fliersize=1, palette=colors[1:], boxprops=dict(alpha=.75), showmeans=True, ax=ax[0,0] )
ax[0,0].set_yticklabels(['Baseline', 'MAR+SID'], rotation=270,
    horizontalalignment='center',
    verticalalignment='center',
    multialignment='center')
# plt.xlim(-0.6,0.4)
# plt.ylabel('RT60 error (s)')
# plt.subplot(2,2,2)
# plt.grid()
sns.boxplot(data=data[1::4], orient='h', fliersize=1, palette=colors[1:], boxprops=dict(alpha=.75), showmeans=True, ax=ax[0,1] )
ax[0,1].set_yticklabels(['', ''], rotation=270,
    horizontalalignment='center',
    verticalalignment='center',
    multialignment='center')
# ax.set_yticklabels(['', ''])
# plt.xlim(-1,3)


 ################### CUMULATIVE HISTOGRAM###################

# plt.subplot(2,2,3)
# plt.grid()
sns.distplot(data[0],hist = True, kde = True, norm_hist=True,
             kde_kws = {'shade': True,'linewidth': 1, "linestyle":'--', "color": colors[1]}, ax=ax[1,0],
             hist_kws={"color": colors[1]},
             label =  methods[0])
sns.distplot(data[4],hist = True, kde = True, norm_hist=True,
             kde_kws = {'shade': True,'linewidth': 1, "linestyle":':', "color": colors[2]}, ax=ax[1,0],
             # hist_kws={"histtype": "step", "linewidth": 1, "linestyle":'--', "alpha": 1, "color": colors[2]},
             hist_kws={"color": colors[2]},
             label = methods[2])
ax[0,0].set_xlim(-0.5,0.3)
ax[1,0].set_xlim(-0.5,0.3)
# plt.xlabel('RT60 error (s)')
# plt.subplot(1,2,2)
# plt.subplot(2,2,4)
# plt.grid()
sns.distplot(data[1],hist = True, kde = True, norm_hist=True,
             kde_kws = {'shade': True,'linewidth': 1, "linestyle":'--', "color": colors[1]}, ax=ax[1,1],
             hist_kws={"color": colors[1]},
             label =  methods[0])
sns.distplot(data[5],hist = True, kde = True, norm_hist=True,
             kde_kws = {'shade': True,'linewidth': 1, "linestyle":':', "color": colors[2]}, ax=ax[1,1],
             # hist_kws={"histtype": "step", "linewidth": 1, "linestyle":'--', "alpha": 1, "color": colors[2]},
             hist_kws={"color": colors[2]},
             label = methods[2])
ax[0,1].set_xlim(-1,1.5)
ax[1,1].set_xlim(-1,1.5)
# plt.xlabel('RT60 error (s)')
plt.legend()

ax[1,0].set_xlabel('RT60 error (s)')
ax[1,1].set_xlabel('RT60 error (s)')
ax[0,0].grid(axis='x')
ax[1,0].grid(axis='x')
ax[0,1].grid(axis='x')
ax[1,1].grid(axis='x')

ax[0,0].yaxis.tick_left()
ax[0,1].yaxis.tick_left()
ax[1,0].yaxis.tick_right()
ax[1,1].yaxis.tick_right()




# %%


### results
# SPEECH
import scipy.stats
# print('mean\t\iqr\t\tmedian')
# print("-------------------")
# print('SPEECH')
# print('baseline', np.nanmean(data[0]), scipy.stats.iqr(data[0]), np.nanmedian(data[0]))
# print('oracle sid', np.nanmean(data[2]), scipy.stats.iqr(data[2]), np.nanmedian(data[2]))
# print('mar+sid', np.nanmean(data[4]), scipy.stats.iqr(data[4]), np.nanmedian(data[4]))
# print("-------------------")
# print('DRUMS')
# print('baseline', np.nanmean(data[1]), scipy.stats.iqr(data[1]), np.nanmedian(data[1]))
# print('oracle sid', np.nanmean(data[3]), scipy.stats.iqr(data[3]), np.nanmedian(data[3]))
# print('mar+sid', np.nanmean(data[5]), scipy.stats.iqr(data[5]), np.nanmedian(data[5]))


### ACE RESULTS



# r = np.empty((C, I, max(N), 3))
# r[0, :,:N[0]] = v_speech[:I]
# r[1, :,:N[1]] = v_drums[:I]



# data = []
# for method_idx, method in enumerate(methods):
#     for inst_idx, inst in enumerate(instruments):
#         rrr =  r[inst_idx, :, :, method_idx].flatten()
#         # exclude nans
#         rrr = rrr[~np.isnan(rrr)]
#         # exlude bigger than threshold
#         rrr = rrr[np.abs(rrr)<th]
#         data.append(rrr[~np.isnan(rrr)])
# data = np.asarray(data)



print('speech')
print('method', 'bias', 'mse', 'rho')
# speech
for method_idx, method in enumerate(methods):
    diff = r_speech[:,:,method_idx+1] - r_speech[:,:,0]
    r0 = r_speech[:,:,0].flatten()
    r1 = r_speech[:,:,method_idx+1].flatten()
    # remove bigger than threshold
    r0 = r0[np.abs(r1 < th)]
    r1 = r1[np.abs(r1<th)]

    plt.figure()
    plt.plot(r0)
    plt.plot(r1)

    # bias: mean
    bias = np.mean(diff)
    # mse
    mse = np.sum(np.power(diff, 2)) / len(diff)
    # pearson corr
    rho, _ = scipy.stats.pearsonr(r_speech[:,:,0].flatten(), r_speech[:,:,method_idx+1].flatten())
    # print
    print(method, bias, mse, rho)

# drums
print('drums')
print('method', 'bias', 'mse', 'rho')
for method_idx, method in enumerate(methods):
    diff = r_drums[:,:,method_idx+1] - r_drums[:,:,0]
    r0 = r_drums[:,:,0].flatten()
    r1 = r_drums[:,:,method_idx+1].flatten()
    # remove bigger than threshold
    r0 = r0[np.abs(r1 < th)]
    r1 = r1[np.abs(r1<th)]

    plt.figure()
    plt.plot(r0)
    plt.plot(r1)

    # bias: mean
    bias = np.mean(diff)
    # mse
    mse = np.sum(np.power(diff, 2)) / len(diff)
    # pearson corr
    rho, _ = scipy.stats.pearsonr(r0, r1)
    # print
    print(method, bias, mse, rho)

