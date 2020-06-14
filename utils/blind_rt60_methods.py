import numpy as np
import scipy.signal, scipy.optimize


def continuously_descendent(array):
    return np.all(array[:-1] > array[1:])

def line_origin(x, m):
    return m * x

def line(x, m, n):
    return m * x + n

def rt60_bands(rt60_0, nBands, decay=0.1):
    # decay per octave
    return np.asarray([rt60_0-(decay*i) for i in range(nBands)])

def herm(A):
    return np.conj(np.transpose(A))

def inner_norm(d, PHI):
    return np.sqrt( herm(d) @ np.linalg.pinv(PHI) @ d )



# %%  Main method by Prego et. al.
def estimate_blind_rt60(r_tf, sr=8000, window_overlap=256, FDR_time_limit=0.5):


    # %%  Subband FDR detection

    K, L = r_tf.shape
    e_tf = np.power(np.abs(r_tf), 2)  # energy spectrogram
    Llim = int(np.ceil(FDR_time_limit / (window_overlap / sr)))  # Number of consecutive windows to span FDR_time_lim

    # PAPER METHOD: at least one DFT for subband
    regions = []
    region_lens = []
    min_num_regions_per_band = 1  # this is per band!
    min_Llim = 3
    for k in range(K):
        num_regions_per_band = 0
        cur_Llim = Llim
        while num_regions_per_band < min_num_regions_per_band and cur_Llim >= min_Llim:
            cur_Llim -= 1
            num_regions_per_band = 0  # per band!
            for l in range(0, L - cur_Llim):
                if continuously_descendent(e_tf[k, l:l + cur_Llim]):
                    regions.append((k, l))
                    region_lens.append(cur_Llim)
                    num_regions_per_band += 1

    num_regions = len(regions)
    if num_regions == 0:
        raise ValueError('No FDR regions found!!')

    # PLOT -----
    # regions_tf = np.empty((K, L))
    # regions_tf.fill(np.nan)
    # for i, (k, l) in enumerate(regions):
    #     ll = region_lens[i]
    #     regions_tf[k, l:l + ll] = ll

    # plt.figure()
    # plt.title('Free Decay Regions')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Frequency (bins)')
    # plt.pcolormesh(20*np.log10(np.abs(r_tf)), vmin = -60, vmax = -20, cmap = 'inferno')
    # plt.pcolormesh(regions_tf)
    # print('number of regions:', num_regions)

    # %% 3. Subband Feature Estimation
    rt60s = np.zeros(num_regions)
    sedf = [[] for n in range(num_regions)]

    for FDR_region_idx, FDR_region in enumerate(regions):
        k, l = FDR_region
        Llim = region_lens[FDR_region_idx]
        den = np.sum(e_tf[k, l:l + Llim])

        for ll in range(Llim):
            num = np.sum(e_tf[k, l + ll:l + Llim])
            if num == 0: # filter out the TF bins with no energy
                Llim -= 1
            else:
                sedf[FDR_region_idx].append(10 * np.log10(num / den))

        m = scipy.optimize.curve_fit(line_origin, np.arange(Llim), sedf[FDR_region_idx])[0][0]
        y2 = -60
        x2 = y2 / m  # in number of windows
        rt60s[FDR_region_idx] = x2 * window_overlap / sr  # in seconds

    # plt.figure()
    # plt.plot(sedf[3], label='True SEDF')
    # plt.grid()
    # plt.plot(np.arange(9), np.arange(9) * m, linestyle='--', label='Linear fit')
    # plt.title('Free Decay Region')
    # plt.xlabel('Time frames')
    # plt.ylabel('Subband Energy Decay Function')
    # plt.legend()
    # decay curves

    # stats of rt60s
    # plt.hist(rt60s, bins = np.arange(0,1,0.05))
    # plt.grid()
    # plt.xlabel('RT60')
    # plt.title('FDR Histogram of estimated RT60s')

    # %% 4. Statistical analysis of subbands RTs

    # Compute the RT(k) as the median of all RT estimates of subband k
    RT_per_subband = [ [] for k in range(K)] # init to empty list of lists„

    for FDR_region_idx, FDR_region in enumerate(regions): # group decay estimates by frequency bin
        k, _ = FDR_region
        RT_per_subband[k].append(rt60s[FDR_region_idx])

    # take the median
    median_RT_per_subband = []
    for k in range(K):
        rts = RT_per_subband[k]
        if len(rts) > 0: # not empty
            median_RT_per_subband.append(np.median(rts))

    # Final value: median of all subband medians
    return np.median(np.asarray(median_RT_per_subband))





# %% MULTICHANNEL AUTORECURSIVE MODEL


##########################################################################################
##### MAR MODELS

# def unwrap_MAR_coefs(c, L):
#     """
#     Take the coefs in vector form (c) as given by dereverberation_* methods,
#     and rearrange them in matrix form (C)
#
#     :param c: matrix, (Lc, dimK, dimN), Lc =  dimM * dimM * L
#     :return: C: matrix(L, dimM, dimM, dimK, dimN). Most recent coefs first (D, D+1... Lar)
#     """
#     # todo assert dims
#
#     Lc, dimK, dimN = c.shape
#     dimM = int(np.sqrt(Lc/L))
#
#     C = np.empty((L, dimM, dimM, dimK, dimN), dtype='complex')  # t, rows, cols
#     for col in range(dimM):
#         col_idx = col * L * dimM
#         for l in range(L):
#             l_idx = l * dimM + col_idx
#             C[L - l - 1, col] = c[l_idx:l_idx + dimM]
#     return C
#
# def apply_MAR_coefs(C, s_tf, L, D, time_average=False):
#     """
#
#     :param C: matrix(L, dimM, dimM, dimK, dimN)
#     :param s_tf:
#     :return:
#     """
#
#     dimM, dimK, dimN = s_tf.shape
#     r_tf = np.zeros(np.shape(s_tf), dtype='complex')
#
#     if time_average: # only take coefficients from D on, although difference is very small (around -70 db for 3 seconds average)
#         C = np.mean(C[:,:,:,:,D:], axis=-1)
#
#     for n in range(D, dimN):
#         for l in range(0, L):
#             if time_average:
#                 a = np.transpose(C[l], (2, 0, 1))  # Expansion at the first dimension (k)
#             else:
#                 a = np.transpose(C[l, :, :, :, n], (2, 0, 1))  # Expansion at the first dimension (k)
#             b = r_tf[:, :, n - (D + l)].T[:, :, np.newaxis]  # Add a dummy dimension for the matmul
#             r_tf[:, :, n] += (a @ b).squeeze().T  # Remove the dummy dimmension
#         r_tf[:, :, n] += s_tf[:, :, n]  # Add novelty signal
#
#     return r_tf
#
#     # non-optimized version
#     # r3_tf = np.zeros(np.shape(s_tf), dtype='complex')
#     # for k in range(dimK):
#     #     # print(k)
#     #     for n in range(D, dimN):
#     #         for l in range(0, L):
#     #             r3_tf[:, k, n] += c[l, :, :, k] @ r3_tf[:, k, n-(D+l)]
#     #         r3_tf[:, k, n] += s2_tf[:, k, n] # current
#     # assert np.allclose(r2_tf,r3_tf)
#
#
# def get_MAR_transition_matrix_eigenvalues(C, time_average=False):
#     """
#
#     https://dsp.stackexchange.com/questions/31859/how-do-i-test-stability-of-a-mimo-system
#
#     :param C: matrix(L, dimM, dimM, dimK, dimN)
#     :return:
#     """
#
#     L, dimM, _, dimK, dimN = C.shape
#
#     if not time_average:
#         e = np.empty((dimM * L, dimK, dimN), dtype=complex)
#
#         for n in range(dimN):
#             A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)
#             # Put C matrices in the first M rows
#             for l in range(L):
#                 A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l, :, :, :, n], (2, 0, 1))  # put k in the first dim for broadcasting
#             # Fill the resting M(L-1) x ML matrix with identity matrices
#             I = np.tile(np.identity(dimM * L)[np.newaxis], (dimK, 1, 1))  # dimK identity matrices in parallel
#             A[:, dimM:, :] = I[:, :-dimM]
#             # Find eigenvalues
#             e[:, :, n] = np.linalg.eigvals(A).T  # (dimM * L, dimK)
#     else:
#         C = np.mean(C, axis=-1)
#         A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)  # Transition matrix
#         # Put C matrices in the first M rows
#         for l in range(L):
#             A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l], (2, 0, 1)) # put k in the first dim for broadcasting
#         # Fill the resting M(L-1) x ML matrix with identity matrices
#         I = np.tile(np.identity(dimM * L)[np.newaxis], (dimK, 1, 1))  # dimK identity matrices in parallel
#         A[:, dimM:, :] = I[:, :-dimM]
#         # Find eigenvalues
#         e = np.linalg.eigvals(A).T  # (dimM * L, dimK)
#     return e
#
#
#
# def build_recursive_matrix(y, n, dimM, L):
#     """
#     operation described in Eq. 6.8
#     :return:
#     """
#
#     # Construct y_Vec
#     y_vec = np.zeros(dimM * L, dtype='complex')  # second term of # 6.8
#     for x in np.arange(L):  # check!
#         nd = L - 1 - x
#         for m in range(dimM):
#             y_vec[x * dimM + m] = y[m, n-nd]
#
#     return np.kron(np.identity(dimM), y_vec)  # 6.8



def estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    X = y_tf.transpose((1,2,0))  # [K, N, M]
    i = 0
    D = X  # [K, N, M]
    PHI = np.tile(np.identity(dimM)[np.newaxis], (dimK, 1, 1)) # [K, M, M]
    F = ita  # just for initialization
    F_k = ita  # just for initialization

    # Get recursive matrix
    Xtau = np.zeros((dimK, dimN, dimM * L), dtype=complex)  # [K, N, ML]
    for m in range(dimM):
        Xtau_m = np.zeros((dimK, dimN, L), dtype=complex) # [K, N, L]
        for l in range(L):
            for n in range(dimN):
                if n >= tau + l:  # avoid aliasing
                    Xtau_m[:, n, l] = X[:, n - tau - l, m]
        Xtau[:, :, L * m:L * (m + 1)] = Xtau_m

    # while i < i_max and F >= ita:
    while i < i_max and np.mean(F_k) >= ita:
        print('  iter',i, 'np.mean(F_k)', np.mean(F_k))

        last_D = D # [K, N, M]

        def herm_k(X):
            return X.conj().transpose((0, 2, 1))

        def transpose_k(X):
            return X.transpose((0, 2, 1))

        # Estimate weights
        w = np.empty((dimK, dimN), dtype='complex')  # [K, N]
        for n in range(dimN):
            d_n = last_D[:, n, :][:, :, np.newaxis]  # [K, N, 1]
            # inner = np.squeeze(np.sqrt(d_n.conj().transpose((0, 2, 1)) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            inner = np.squeeze(np.sqrt(herm_k(d_n) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            w[:, n] = np.power(np.power(inner, 2) + epsilon, (p / 2) - 1)

        # Estimate G
        # todo parallelize
        W = np.empty((dimK, dimN, dimN), dtype=complex) # [K, N, N]
        for k in range(dimK):
            W[k] = np.diag(w[k])

        G = np.linalg.pinv(herm_k(Xtau) @ W @ Xtau) @ (herm_k(Xtau) @ W @ X)  # [K, ML, M]

        # Estimate D
        D = X - (Xtau @ G) # [K, N, M]

        # Estimate PHI
        PHI = (1 / dimN) * (transpose_k(D) @ W @ D.conj()) # [K, M, M]

        # Estimate convergence
        # F = np.linalg.norm(D - last_D) / np.linalg.norm(D)
        # Per-band convergence
        F_k =  np.linalg.norm(D - last_D, axis=(1,2)) / np.linalg.norm(D, axis=(1,2))
        # print(F_k < ita_k)
        # print(F_k)
        # plt.figure()
        # plt.title(str(i))
        # plt.plot(F_k)
        # plt.hlines(np.mean(F_k), 0,dimK-1)
        # plt.show()
        # print(np.mean(F_k), F)

        # Update pointer
        i += 1

    return D.transpose((2, 0, 1)), G, PHI



def estimate_MAR_sparse_oracle(y_tf, s_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    oracle SCM
    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
    C = np.empty((dimK, dimM*L, dimM), dtype=complex)

    phi_oracle = np.empty((dimK, dimM, dimM), dtype=complex)
    for k in range(dimK):
        s = s_tf[:, k, :].T
        phi_oracle[k] = herm(s) @ s

    for k in range(dimK):
        print(k)
        X = y_tf[:, k, :].T  # [N, M]
        i = 0
        D = X # [N, M]
        F = ita # just for initialization

        # Get recursive matrix
        Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
        for m in range(dimM):
            Xtau_m = np.zeros((dimN, L), dtype=complex)
            for l in range(L):
                for n in range(dimN):
                    if n >= tau + l :  # avoid aliasing
                        Xtau_m[n, l] = X[n - tau - l, m]
            Xtau[:, L * m:L * (m + 1)] = Xtau_m

        while i < i_max and F >= ita:
            # print('  iter',i, 'F', F)
            last_D = D

            # Estimate weights
            w = np.empty(dimN, dtype='complex') # probably real...
            for n in range(dimN):
                d_n = last_D[n,:][:,np.newaxis]
                w[n] = np.power( np.power(inner_norm(d_n, phi_oracle[k]), 2) + epsilon, (p/2)-1)

            # Estimate G
            W = np.diag(w) # [N, N]
            G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

            # Estimate D
            D = X - ( Xtau @ G )

            # Estimate convergence
            F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

            # Update pointer
            i += 1

        # Assign
        est_s_tf[:, k, :] = D.T
        C[k, :, :] = G

    return est_s_tf, C


def estimate_MAR_sparse_identity(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    oracle SCM
    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
    C = np.empty((dimK, dimM*L, dimM), dtype=complex)

    # phi as identity matrix
    phi = np.empty((dimK, dimM, dimM), dtype=complex)
    for k in range(dimK):
        phi[k] = np.identity(dimM)

    for k in range(dimK):
        print(k)
        X = y_tf[:, k, :].T  # [N, M]
        i = 0
        D = X # [N, M]
        F = ita # just for initialization

        # Get recursive matrix
        Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
        for m in range(dimM):
            Xtau_m = np.zeros((dimN, L), dtype=complex)
            for l in range(L):
                for n in range(dimN):
                    if n >= tau + l :  # avoid aliasing
                        Xtau_m[n, l] = X[n - tau - l, m]
            Xtau[:, L * m:L * (m + 1)] = Xtau_m

        while i < i_max and F >= ita:
            # print('  iter',i, 'F', F)
            last_D = D

            # Estimate weights
            w = np.empty(dimN, dtype='complex') # probably real...
            for n in range(dimN):
                d_n = last_D[n,:][:,np.newaxis]
                w[n] = np.power( np.power(inner_norm(d_n, phi[k]), 2) + epsilon, (p/2)-1)

            # Estimate G
            W = np.diag(w) # [N, N]
            G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

            # Estimate D
            D = X - ( Xtau @ G )

            # Estimate convergence
            F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

            # Update pointer
            i += 1

        # Assign
        est_s_tf[:, k, :] = D.T
        C[k, :, :] = G

    return est_s_tf, C
