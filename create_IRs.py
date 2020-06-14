"""
create_IRs.py
Create a set of IRs with the given specs,
and save them to a folder, together with the associated metadata.
"""

import numpy as np
import soundfile as sf
import os.path
import csv

import masp
from masp import shoebox_room_sim as srs


# %% GLOBAL SETUP - MUST MATCH ACROSS FILES!

fs = 8000
sh_order = 1
dimM = (sh_order+1)**2

# Number of iterations
I = 100

# IR frequency band
rt60_f = np.asarray([1000])

# Destination IRs folder
IR_folder_path = '/Users/andres.perez/source/ambisonic_rt_estimation/IRs'
if not os.path.exists(IR_folder_path):
    os.mkdir(IR_folder_path)

# %% ROOM SETUP

# m*x + n
rt60_m = 0.6
rt60_n = 0.4

# frequency band
rt60_f = np.asarray([1000])

# Fixed room size
room = np.array([10.2, 7.1, 3.2])

# Receiver at the center of the room
rec = (room/2)[np.newaxis]
nRec = rec.shape[0]

# SH orders for receivers
rec_orders = np.array([sh_order])

# maximum IR length in seconds
maxlim = 1.
limits = np.asarray([maxlim])


# %% ITERATION: IR rendering

for i in range(I):
    print('--------------------------------------------')
    print('ITER:', i)
    
    # Random rt60
    rt60 = np.random.rand() * rt60_m + rt60_n
    print('RT60:', rt60)

    # Critical distance for the room
    abs_wall = srs.find_abs_coeffs_from_rt(room, np.asarray([rt60]))[0]
    _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)

    # Random source position, half critical distance
    azi = np.random.rand() * 2 * np.pi
    incl = np.random.rand() * np.pi
    azi = azi + np.pi # TODO: fix in srs library!!!
    src_sph = np.array([azi, np.pi/2-incl, d_critical.mean()/2])
    src_cart = masp.sph2cart(src_sph)
    src = rec + src_cart
    nSrc = src.shape[0]

    # Render echogram
    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, rt60_f, fs).squeeze().T
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)

    # Write audio file
    audio_file_name = str(i)+'.wav'
    audio_file_path = os.path.join(IR_folder_path, audio_file_name)
    sf.write(audio_file_path, irs.T, samplerate=fs)

    # Write metadata file
    metadata_file_name = str(i)+'.csv'
    metadata_file_path = os.path.join(IR_folder_path, metadata_file_name)
    with open(metadata_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["azi", (azi - np.pi)*360/(2*np.pi)])
        writer.writerow(["ele", (np.pi/2 - incl)*360/(2*np.pi)])
        writer.writerow(["dist", d_critical.mean()/2])
        writer.writerow(["rt60", rt60])
