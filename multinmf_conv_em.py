import numpy as np
import numpy.random as random

import pyroomacoustics as pra
from multinmf_recons_im import multinmf_recons_im

def multinmf_conv_em(X, W0, H0, A0, Sigma_b0, source_NMF_ind, iter_num=100,
        SimAnneal_flag=0, Sigma_b_Upd_flag=False, update_a=True, update_w=True, update_h=True, verbose=False):

    # [W,H,A,Sigma_b,S,log_like_arr] = ...
    #    multinmf_conv_em(X, W0, H0, A0, Sigma_b0, source_NMF_ind, iter_num, SimAnneal_flag, Sigma_b_Upd_flag);

    # EM algorithm for multichannel NMF decomposition in convolutive mixture (stereo)

    # input
    # -----

    # X                 : truncated STFT of multichannal mixture [F x N x 2]
    # W0                : initial matrix of bases [F x K]
    # H0                : initial matrix of contributions [K x N]
    # A0                : initial (complex-valued) convolutive mixing matrices [F x 2 x J]
    # Sigma_b0          : initial additive noise covariances [F x 1] vector
    # source_NMF_ind    : source indices in the NMF decomposition
    # iter_num          : (opt) number of EM iterations (def = 100)
    # SimAnneal_flag    : (opt) simulated annealing flag (def = 1)
    #                         0 = no annealing
    #                         1 = annealing
    #                         2 = annealing with noise injection
    # Sigma_b_Upd_flag  : (opt) Sigma_b update flag (def = 0)

    # output
    # ------

    # W                 : estimated matrix of bases [F x K]
    # H                 : estimated matrix of contributions [K x N]
    # A                 : estimated (complex-valued) convolutive mixing matrices [F x 2 x J]
    # Sigma_b           : final additive noise covariances [F x 1] vector
    # S                 : estimated truncated STFTs of sources [F x N x J]
    # log_like_arr      : array of log-likelihoods

    ###########################################################################
    # Copyright 2017 Diego Di Carlo, Robin Scheibler
    #   Port to Python, extension to arbitrary number of channels

    # Adapted from Matlab code by Alexey Ozerov
    # Copyright 2010 Alexey Ozerov
    # (alexey.ozerov -at- irisa.fr)

    # This software is distributed under the terms of the GNU Public License
    # version 3 (http://www.gnu.org/licenses/gpl.txt)

    # If you use this code please cite this paper

    # A. Ozerov and C. Fevotte,
    # "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
    # IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
    # of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
    # Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
    ###########################################################################

    # some constants
    final_ann_noise_var = 3e-11
    log_like_threshold = -1e-3#1e-2

    # some variables
    log_like_diff_prev3 = 0
    log_like_diff_prev2 = 0
    log_like_diff_prev = 0

    F, N, I = X.shape
    K = W0.shape[1]
    J = len(source_NMF_ind)

    if verbose:
        print('Multichannel NMF dimension:', F, N, I, J, K)

    # if I != 2:
    #     raise ValueError('Multi_NMF_EM_conv: number of channels must be 2')

    if SimAnneal_flag > 0 and Sigma_b_Upd_flag:
        raise ValueError('The flags SimAnneal_flag and Sigma_b_Upd_flag cannot be on simultaneously')

    Xb = X.copy()
    W = W0.copy()
    H = H0.copy()
    A = A0.copy()
    Sigma_b = Sigma_b0.copy()

    W_prev = np.empty_like(W)
    H_prev = np.empty_like(H)
    A_prev = np.empty_like(A)
    Sigma_b_prev = np.empty_like(Sigma_b)

    O = np.ones(N)

    sigma_ss = np.zeros((F, N, J))
    Sigma_x = np.zeros((F, N, I, I), dtype=np.complex)
    Inv_Sigma_x = np.zeros((F, N, I, I), dtype=np.complex)
    Gs = np.zeros((F, N, J, I), dtype=np.complex)
    Gs_x = np.zeros((F, N, J), dtype=np.complex)
    bar_Rxs = np.zeros((F, I, J), dtype=np.complex)
    bar_Rss = np.zeros((F, J, J), dtype=np.complex)
    bar_Rxx = np.zeros((F, I, I), dtype=np.complex)
    bar_A = np.zeros((F, I, K), dtype=np.complex)
    Vc = np.zeros((F, N, K))
    log_like_arr = np.zeros((iter_num))

    # Normalization of A
    for j in range(J):
        nonzero_f_ind = np.where(A[:,0,j] != 0)[0]
        sign = A[nonzero_f_ind,0,j] / np.abs(A[nonzero_f_ind,0,j])
        for i in range(I):
            A[nonzero_f_ind,I-i-1,j] = A[nonzero_f_ind,I-i-1,j] / sign

        A_scale = np.sum(np.abs(A[:,:,j]) ** 2, axis = 1)
        A[:,:,j] = A[:,:,j] / np.sqrt(A_scale[:,None])
        W[:,source_NMF_ind[j]] = W[:, source_NMF_ind[j]] * A_scale[:,None]

    # initialize simulated annealing variances (if necessary)
    if SimAnneal_flag > 0:
        Sigma_b_anneal = np.zeros((F, iter_num))
        for iter in range(iter_num):
            Sigma_b_anneal[:, iter] = ((np.sqrt(Sigma_b0) * (iter_num - iter) \
                + np.ones(F) * np.sqrt(final_ann_noise_var) * iter) / iter_num) ** 2

    # avoiding divergence
    W[np.where(W<=1e-6)[0]]=1e-6

    # MAIN LOOP
    for iter in range(iter_num):
        if verbose:
            print('EM iteration {} of {}:'.format(iter, iter_num))

        # store parameters estimated on previous iteration
        np.copyto(W_prev, W)
        np.copyto(H_prev, H)
        np.copyto(A_prev, A)
        np.copyto(Sigma_b_prev, Sigma_b)

        # E-step: compute expectations of natural suffitient statistics
        if verbose:
            print('   - E-step')

        # compute a priori source variances
        sigma_ss[:,] = 0
        for j in range(J):
            sigma_ss[:,:,j] = np.dot(W[:,source_NMF_ind[j]], H[source_NMF_ind[j],:])

        if SimAnneal_flag:
            Sigma_b = Sigma_b_anneal[:,iter]

            if SimAnneal_flag == 2:  # with noise injection
                Noise = np.random.randn(F, N, I) + 1j * np.random.randn(F, N, I)  # complex noise
                for i in range(I):
                    Noise[:,:,i] *= np.outer(np.sqrt(Sigma_b / 2), np.ones(N))

                Xb = X + Noise

        # compute the Sigma_x matrix (generalized to any number of channels)
        for i in range(I):
            for ii in range(i,I):
                cc = A[:,i,:] * np.conj(A[:,ii,:])
                Sigma_x[:,:,i,ii] = np.sum(cc[:,None,:] * sigma_ss, axis=2)
                if i != ii:
                    Sigma_x[:,:,ii,i] = np.conj(Sigma_x[:,:,i,ii])
                else:
                    Sigma_x[:,:,i,ii] += Sigma_b[:,None]

        # compute the inverse of Sigma_x matrix
        Det_Sigma_x = np.real(np.linalg.det(Sigma_x))
        Inv_Sigma_x = np.linalg.inv(Sigma_x)

        # compute log-likelihood
        xS = np.matmul(np.conj(Xb[:,:,None,:]), Inv_Sigma_x)
        xSx = np.real(np.matmul(xS, Xb[:,:,:,None]))

        log_like = - np.sum( np.squeeze(xSx) + np.log(Det_Sigma_x * np.pi)) / (N * F)

        if iter > 1:
            log_like_diff = log_like - log_like_arr[iter-1]
            if verbose:
                print('      Log-likelihood: {}\n      Log-likelihood improvement: {}'.format(log_like, log_like_diff))

            # if the increment of the log-likelihood is less, exit
            if iter > 10:
                if SimAnneal_flag>1 and\
                (log_like_diff<0 and log_like_diff_prev<0 \
                    and log_like_diff_prev2<0):
                    break
                elif SimAnneal_flag==0 and log_like_diff < log_like_threshold:
                    break

            log_like_diff_prev2 = log_like_diff_prev
            log_like_diff_prev = log_like_diff
        else:
            if verbose:
                print('      Log-likelihood:', log_like)
        log_like_arr[iter] = log_like

        for j in range(J):
            # compute S-Wiener gain
            Gs[:,:,j,:] = np.einsum('fti,ftiI->ftI',
                            np.einsum('ft,fi->fti', sigma_ss[:,:,j], np.conj(A[:,:,j])),
                            Inv_Sigma_x)
            # compute Gs_x
            Gs_x[:,:,j] = np.einsum('fti,fti->ft', Gs[:,:,j,:], Xb)

            # compute average Rxs
            for i in range(I):
                bar_Rxs[:,i,j] = np.mean(Xb[:,:,i] * np.conj(Gs_x[:,:,j]), axis=1)

        for j1 in range(J):
            # compute average Rss
            for j2 in range(J):
                GsA = np.einsum('fti,fi->ft',Gs[:,:,j1,:],A[:,:,j2])
                bar_Rss[:,j1,j2] = np.mean(
                        Gs_x[:,:,j1] * np.conj(Gs_x[:,:,j2])
                        - GsA * sigma_ss[:,:,j2],
                                   axis=1)
            bar_Rss[:,j1,j1] = bar_Rss[:,j1,j1] + np.mean(sigma_ss[:,:,j1], axis=1)

        # compute average Rxx
        bar_Rxx = np.matmul(np.moveaxis(Xb, [-1], [-2]), np.conj(Xb)) / Xb.shape[1]


        # TO ASSURE that Rss = Rss'
        for f in range(F):
            bar_Rss[f,:,:] = (bar_Rss[f,:,:] + np.conj(bar_Rss[f,:,:].T)) / 2

        # compute extended mixing matrix A
        for j in range(J):
            for k in source_NMF_ind[j]:
                bar_A[:,:,k] = A[:,:,j]

        for k in range(K):
            # compute a priori component variances
            sigma_cc_k = np.dot(W[:,k,None], H[None,k,:])

            # compute C-Wiener gain
            Gc_k = ( np.einsum('fi,ftil->ftl', np.conj(bar_A[:,:,k]), Inv_Sigma_x)\
                 * sigma_cc_k[:,:,None])

            # compute Gc_x
            Gc_x_k = np.einsum('fni,fni->fn', Gc_k, Xb)

            # compute components sufficient natural statistics
            # IT IS IMPORTANT TO TAKE A REAL PART !!!!
            Vc[:,:,k] = np.abs(Gc_x_k) ** 2 + sigma_cc_k \
                        - np.real(np.einsum('fti,fi->ft',Gc_k, bar_A[:,:,k])) \
                            * sigma_cc_k

        # M-step: re-estimate
        if verbose:
            print('   - M-step')

        # re-estimate A
        if update_a:
            if verbose:
                print("     - Update A")
            for f in range(F):
                A[f,:,:] = np.dot(bar_Rxs[f,:,:], np.linalg.inv(bar_Rss[f,:,:]))

        # re-estimate noise variances (if necessary)
        if Sigma_b_Upd_flag:
            for f in range(F):
                Sigma_b[f] = 0.5 * np.real(
                        np.trace(
                            bar_Rxx[f, :, :] -
                            np.dot(A[f,:,:], np.conj(bar_Rxs[f, :, :].T)) -
                            np.dot(bar_Rxs[f,:,:], np.conj(A[f, :, :].T)) +
                            np.dot(np.dot(A[f, :, :], bar_Rss[f, :, :]), np.conj(A[f, :, :].T))
                            )
                        )
        # re-estimate W, and then H
        for k in range(K):
            if update_w:
                if not k and verbose: 
                    print("     - Update W") # print just the first one
                W[:, k] = np.sum(Vc[:,:,k] / np.outer(np.ones(F), H[k, :]), axis=1) / N
            if update_h:
                if not k and verbose:
                    print("     - Update H\n")
                H[k, :] = np.sum(Vc[:,:,k] / np.outer(W[:,k], np.ones(N)), axis=0) / F

        # Normalization of A
        if update_a:
            for j in range(J):
                nonzero_f_ind = np.where(A[:,0,j] != 0)[0]
                sign = A[nonzero_f_ind,0,j] / np.abs(A[nonzero_f_ind,0,j])
                for i in range(I):
                    A[nonzero_f_ind,I-i-1,j] = A[nonzero_f_ind,I-i-1,j] / sign

                A_scale = np.sum(np.abs(A[:,:,j]) ** 2, axis = 1)
                A[:,:,j] = A[:,:,j] / np.sqrt(A_scale[:,None])
                if update_w:
                    W[:,source_NMF_ind[j]] = W[:, source_NMF_ind[j]] * A_scale[:,None]

        # Normalisation of W components
        if update_w:
            w = np.sum(W, axis=0)
            W /= w[np.newaxis,:]
            H *= w[:,np.newaxis]  # Energy transfer to H

    # source estimates
    S = Gs_x

    # return parameters estimated on previous iteration, since they were used
    # for source estimates computation
    W = W_prev
    H = H_prev
    A = A_prev
    Sigma_b = Sigma_b_prev

    return W, H, A, Sigma_b, S, log_like_arr


def multinmf_conv_em_wrapper(
        x, n_src, stft_win_len, n_latent_var, n_iter=500, \
        A_init=None, W_init=None, H_init=None, \
        update_a=True, update_w=True, update_h=True, \
        verbose = False):

    '''
    A wrapper around multichannel nmf using EM updates to use with pyroormacoustcs.
    Performs STFT and ensures all signals are the correct shape.

    Parameters
    ----------
    x: ndarray
        (n_samples x n_chan) array of time domain samples
    n_latent_var: int
        number of latent variables in the NMF
    '''

    n_chan = x.shape[1]

    # STFT
    window = np.sqrt(pra.cosine(stft_win_len))  # use sqrt because of synthesis
    # X is (n_chan, n_frame, n_bin)
    X = np.array(
            [pra.stft(x[:,ch], stft_win_len, stft_win_len // 2, win=window, transform=np.fft.rfft) for ch in range(n_chan)]
            )
    # move axes to match Ozerov's order (n_bin, n_frame, n_chan)
    X = np.moveaxis(X, [0,1,2], [2,1,0])
    n_bin = X.shape[0]
    n_frame = X.shape[1]

    if W_init is None:
        K = n_latent_var * n_src
    else:
        K = W_init.shape[-1]

    # Random initialization of multichannel NMF parameters
    source_NMF_ind = []
    for j in range(n_src):
        source_NMF_ind = np.reshape(np.arange(K, dtype=np.int), (n_src,-1))

    mix_psd = 0.5 * (np.mean(np.sum(np.abs(X)**2, axis=2), axis=1))
    if A_init is None:
        # random initialization
        update_a = True
        A_init = (0.5 *
                    ( 1.9 * np.abs(random.randn(n_bin, n_chan, n_src))       \
                    + 0.1 * np.ones((n_bin, n_chan, n_src))                  \
                    ) * np.sign( random.randn(n_bin, n_chan, n_src)          \
                                + 1j * random.randn(n_bin, n_chan, n_src))  \
                )
    else:
        # reshape the partial rir input (n_bin, n_chan, n_src)
        A_init = np.moveaxis(A_init, [2], [0])

    # W is intialized so that its enegy follows mixture PSD
    if W_init is None:
        W_init = 0.5 * (
                ( np.abs(np.random.randn(n_bin,K)) + np.ones((n_bin,K)) )
                * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
                )
    if H_init is None:
        H_init = 0.5 * ( np.abs(np.random.randn(K,n_frame)) + np.ones((K,n_frame)) )

    Sigma_b_init = mix_psd / 100


    W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr = \
        multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind,
            iter_num=n_iter, update_a=update_a, update_w=update_w, update_h=update_h, verbose=verbose)

    Ae_EM = np.moveaxis(Ae_EM, [0], [2])

    # Computation of the spatial source images
    if verbose:
        print('Computation of the spatial source images\n')
    Ie_EM = np.zeros((n_bin,n_frame,n_src,n_chan), dtype=np.complex)
    for j in range(n_src):
        for f in range(n_bin):
            Ie_EM[f,:,j,:] = np.outer(Se_EM[f,:,j], Ae_EM[:,j,f])

    sep_sources = []

    # Inverse STFT
    ie_EM = []
    for j in range(n_src):
        # channel-wise istft with synthesis window
        ie_EM = []
        for ch in range(n_chan):
            ie_EM.append(
                    pra.istft(Ie_EM[:,:,j,ch].T, stft_win_len, stft_win_len // 2, win=window, transform=np.fft.irfft)
                    )
        sep_sources.append(np.array(ie_EM).T)

    return np.array(sep_sources)


def multinmf_conv_em_dictionary_training(X, n_latent_var, n_iter):
    '''
    A wrapper around multichannel nmf using EM updates to use with pyroormacoustcs.
    for training dictionary

    Parameters
    ----------
    X: ndarray
        (n_samples x n_chan) array of time domain samples
    W: ndarray
        (n_samples x n_components)
    H: ndarray
        (n_components x n_features)
    n_latent_var: int
        number of latent variables in the NMF
    n_iter: int
        number of iterations
    '''

    n_src = 1 #this wrapper is usually used in a for-loop, learning one source at a time
    n_bin = X.shape[0]
    n_frame = X.shape[1]

    try:
        n_chan = X.shape[2]
    except:
        n_chan = 1;
        X = X[:,:,None]

    # Random initialization of multichannel NMF parameters
    K = n_latent_var * n_src
    source_NMF_ind = []
    for j in range(n_src):
        source_NMF_ind = np.reshape(np.arange(n_latent_var * n_src, dtype=np.int), (n_src,-1))

    mix_psd = 0.5 * (np.mean(
                np.sum(np.abs(X)**2, axis=2),
                axis=1))
    A_init = (0.5 *
                ( 1.9 * np.abs(random.randn(n_bin, n_chan, n_src))       \
                + 0.1 * np.ones((n_bin, n_chan, n_src))                  \
                ) * np.sign( random.randn(n_bin, n_chan, n_src)          \
                            + 1j * random.randn(n_bin, n_chan, n_src))  \
            )
    # W is intialized so that its enegy follows mixture PSD
    W_init = 0.5 * (
            ( np.abs(np.random.randn(n_bin,K)) + np.ones((n_bin,K)) )
            * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
            )
    H_init = 0.5 * ( np.abs(np.random.randn(K,n_frame)) + np.ones((K,n_frame)) )
    Sigma_b_init = mix_psd / 100

    W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr = \
        multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind, iter_num=n_iter)

    return W_EM
