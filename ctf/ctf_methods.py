"""
ctf_method.py
Methods for the Convolutive Transfer Function
"""

# %%
import numpy as np
import os
import soundfile as sf
import tempfile
import matlab.engine
import scipy.signal
import matplotlib.pyplot as plt

# Set to your path
matlab_path = '/Users/andres.perez/source/ambisonic_rt_estimation/ctf/akis_ctfconv'

# Init matlab
eng = matlab.engine.start_matlab()
eng.addpath(matlab_path)


def sid_stft2(x, y, winsize, hopsize, filtersize):
    ml_args = []

    if x.ndim == 1:
        x = x[:, np.newaxis]
    ml_args.append(matlab.double(x.tolist()))

    if y.ndim == 1:
        y = y[:, np.newaxis]
    ml_args.append(matlab.double(y.tolist()))

    ml_args.append(float(winsize))
    ml_args.append(float(hopsize))
    ml_args.append(float(filtersize))
    # if fftsize is not None:
    #     ml_args.append(float(fftsize))
    # if hopsize is not None:
    #     ml_args.append(float(hopsize))
    # if winvec is not None:
    #     ml_args.append(matlab.double(winvec.tolist()))

    ml_method = 'sid_stft2'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()

def ctfconv(inspec, irspec, winsize=None):
    ml_args = []
    ml_args.append(matlab.double(inspec.tolist(), is_complex=True))
    ml_args.append(matlab.double(irspec.tolist(), is_complex=True))
    if winsize is not None:
        ml_args.append(winsize)

    ml_method = 'ctfconv'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()

def fftconv(h, x):
    ml_args = []
    ml_args.append(matlab.double(h.tolist()))
    ml_args.append(matlab.double(x.tolist()))

    ml_method = 'fftconv'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()

def fftpartconv(h, x, L):
    ml_args = []
    ml_args.append(matlab.double(h.tolist()))
    ml_args.append(matlab.double(x.tolist()))
    ml_args.append(L)

    ml_method = 'fftpartconv'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()

def istft(inspec, winsize=None, hopsize=None, lSig=None):
    ml_args = []
    ml_args.append(matlab.double(inspec.tolist(), is_complex=True))
    if winsize is not None:
        ml_args.append(float(winsize))
    if hopsize is not None:
        ml_args.append(float(hopsize))
    if lSig is not None:
        ml_args.append(float(lSig))

    ml_method = 'istft'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()

def stft(insig, winsize, fftsize=None, hopsize=None, winvec=None):
    ml_args = []

    if insig.ndim == 1:
        insig = insig[:, np.newaxis]
    ml_args.append(matlab.double(insig.tolist()))

    ml_args.append(float(winsize))
    if fftsize is not None:
        ml_args.append(float(fftsize))
    if hopsize is not None:
        ml_args.append(float(hopsize))
    if winvec is not None:
        ml_args.append(matlab.double(winvec.tolist()))

    ml_method = 'stft'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()


def sid_stft2(x, y, winsize, hopsize, filtersize):
    ml_args = []

    if x.ndim == 1:
        x = x[:, np.newaxis]
    ml_args.append(matlab.double(x.tolist()))

    if y.ndim == 1:
        y = y[:, np.newaxis]
    ml_args.append(matlab.double(y.tolist()))

    ml_args.append(float(winsize))
    ml_args.append(float(hopsize))
    ml_args.append(float(filtersize))

    ml_method = 'sid_stft2'
    nargout = 1
    ml_res = getattr(eng, ml_method)(*ml_args, nargout=nargout)
    return np.asarray(ml_res).squeeze()


def rt60_bands(rt60_0, nBands, decay=0.1):
    # decay per octave
    # return np.asarray([rt60_0-(rt60_0*decay*i) for i in range(nBands)])
    return np.asarray([rt60_0 - (decay * i) for i in range(nBands)])

#
# def compute_t60_acoustics(ir, fs, bands, rt='t30'):
#
#     # write tmp wav file
#     dirpath = tempfile.mkdtemp()
#     name = 'ir.wav'
#     filename = os.path.join(dirpath, name)
#     sf.write(filename, ir, fs)
#
#     # compute with given method
#     import acoustics
#     t60 = acoustics.room.t60_impulse(filename, bands, rt)
#
#     # delete tmp file
#     os.remove(filename)
#
#     return t60

def compute_t60(ir, fs, bands, plot=False, title=None):
    # https://dsp.stackexchange.com/questions/17121/calculation-of-reverberation-time-rt60-from-the-impulse-response

    rt_methods = ['edt', 't10', 't20', 't30']
    rt60 = np.empty((len(bands), len(rt_methods)))

    # first, remove the pre-delay and start from the pea
    ir = ir[np.argmax(np.abs(ir)):]
    # ir = ir[20:] # TODO: AFTERPEAK

    # filter in frequency bands
    # for the moment just octave bands
    for nb, band in enumerate(bands):
        # bandpass the signal
        flim = [band/np.sqrt(2), band*np.sqrt(2)]
        sos = scipy.signal.butter(3, flim, btype='bandpass', output='sos', fs=fs)
        filtered_ir = scipy.signal.sosfilt(sos, ir)

        # get envelope by hilbert trasform
        a_t = np.abs(scipy.signal.hilbert(filtered_ir))

        # Schroeder integration
        sch = np.cumsum(a_t[::-1] ** 2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))
        # plt.plot(sch_db, label=band)

        # TODO: INCLUDE HERE LUNDEBY'S METHOD AS OPTIONAL ARGUMENT

        def index_of_last_element(array, cond):
            return np.nonzero(array > cond)[0][-1]

        if plot:
            plt.figure()
            plt.title(title)
            # plt.plot(20.0 * np.log10(np.abs(ir)))
            plt.plot(20.0 * np.log10(np.abs(filtered_ir)), label='filtered IR')
            plt.plot(20.0 * np.log10(a_t), label='hilbert', linestyle='--', linewidth=1)
            plt.plot(sch_db, label='schroeder')

        for rt_method_idx, rt_method in enumerate(rt_methods):
            if rt_method == 'edt':
                m = -10 / index_of_last_element(sch_db, -10)  # dB / sample
                n = 0 #offset
            elif rt_method == 't10':
                a = index_of_last_element(sch_db, -5)
                b = index_of_last_element(sch_db, -15)
                m = (15 - 5) / (a - b)
                n = -5 - m * a  # - n = mx + y
            elif rt_method == 't20':
                a = index_of_last_element(sch_db, -5)
                b = index_of_last_element(sch_db, -25)
                m = (25 - 5) / (a - b)
                n = -5 - m * a  # - n = mx + y
            elif rt_method == 't30':
                a = index_of_last_element(sch_db, -5)
                b = index_of_last_element(sch_db, -35)
                m = (35 - 5) / (a - b)
                n = -5 - m * a  # - n = mx + y

            rt60_value = (-60 - n) / m / fs
            rt60[nb, rt_method_idx] = rt60_value

            if plot:
                x = np.arange(len(filtered_ir))
                plt.plot(x, m*x+n, label=rt_method, linestyle='--', linewidth=1)
                plt.legend()

    return rt60

#
# def lundeby(x, plot=False):
#
#     def line_origin(x, m):
#         return m * x
#
#     def line_minus5(x, m):
#         return m * x - 5
#
#     def line(x, m, n):
#         return m * x + n
#
#     # x = sch_db # todo: x should be the schroeder integrated version
#
#     L = len(x)
#     # 1. average: already done
#     # 2. last 10% of the impulse response
#     x_bg = int(np.floor(L*0.9)) # preliminar truncation point
#     min_delta_t = 100
#     delta_t = min_delta_t +1  # whatever
#     max_iter = 21
#     i = 0
#     diff_dB = 5
#     while (np.abs(delta_t) > min_delta_t) and (i < max_iter) and (x_bg < L):
#
#         bg_noise_level = x[x_bg]
#
#         # 3. linear regression between 0dB and bg_noise_level+5dB
#         x_last_idx = len(x[x>(bg_noise_level+diff_dB)])-1 # this index included!
#         # if x_last_idx <= 0:
#         #     break
#
#         x_first_idx = len(x[x>-5]) # this index also included
#
#         y_last_idx = x[x_last_idx]
#         y_first_idx = x[x_first_idx]
#
#         #4. crosspoint at the intersection with bg_noise_level
#
#         xxx = np.arange(x_first_idx, x_last_idx+1)
#         yyy = x[xxx]
#
#         m, n = scipy.optimize.curve_fit(line, xxx, yyy)[0]
#         x_cross = int(np.floor((bg_noise_level-n)/m))
#         # m = scipy.optimize.curve_fit(line_minus5, xxx, yyy)[0][0]
#         # x_cross = int(np.floor((bg_noise_level+5)/m))
#
#         if plot:
#             plt.figure()
#             plt.plot(x)
#             plt.title('iter: '+ str(i))
#             plt.vlines(x_bg, min(x), max(x))
#             plt.hlines(bg_noise_level, 0, L - 1)
#             plt.plot(xxx, yyy)
#             plt.scatter(x_bg, bg_noise_level)
#             # plt.plot(m * xxx)
#             plt.plot(m * xxx +n)
#             plt.scatter(x_cross, bg_noise_level) # intersection point
#
#         delta_t = x_bg - x_cross
#         # # print(i, delta_t, x_bg, m, n)
#         # print(i, delta_t, x_bg, m)
#         #
#         # # 5. advance
#         # x_bg = int(np.floor(x_bg - delta_t))
#         # # x_bg = int(np.floor(x_bg - delta_t))
#         # # x_bg = x_bg - 1
#         # i += 1
#
#     # TODO: COMPLETE IT
#     # return x[:x_bg+1]
#
