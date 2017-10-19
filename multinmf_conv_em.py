import numpy as np
import numpy.random as random

import pyroomacoustics as pra
from multinmf_recons_im import multinmf_recons_im

def multinmf_conv_em(X, W0, H0, A0, Sigma_b0, source_NMF_ind, iter_num=100, SimAnneal_flag=1, Sigma_b_Upd_flag=False):

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
    # Copyright 2017 Robin Scheibler

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

    F, N, I = X.shape
    K = W0.shape[1]
    J = len(source_NMF_ind)

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

    # initialize simulated annealing variances (if necessary)
    if SimAnneal_flag > 0:
        Sigma_b_anneal = np.zeros((F, iter_num))
        for iter in range(iter_num):
            Sigma_b_anneal[:, iter] = ((np.sqrt(Sigma_b0) * (iter_num - iter) \
                + np.ones(F) * np.sqrt(final_ann_noise_var) * iter) / iter_num) ** 2

    # MAIN LOOP
    for iter in range(iter_num):
        print('EM iteration {} of {}\n'.format(iter, iter_num))

        # store parameters estimated on previous iteration
        np.copyto(W_prev, W)
        np.copyto(H_prev, H)
        np.copyto(A_prev, A)
        np.copyto(Sigma_b_prev, Sigma_b)

        # E-step: compute expectations of natural suffitient statistics
        print('   E-step\n')

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
                Sigma_x[:,:,i,ii] = np.sum(cc[:,np.newaxis,:] * sigma_ss, axis=2)
                if i != ii:
                    Sigma_x[:,:,ii,i] = np.conj(Sigma_x[:,:,i,ii])
                else:
                    Sigma_x[:,:,i,ii] += Sigma_b[:,np.newaxis]

        # compute the inverse of Sigma_x matrix
        Det_Sigma_x = np.real(np.linalg.det(Sigma_x))
        Inv_Sigma_x = np.linalg.inv(Sigma_x)

        # compute log-likelihood
        xS = np.matmul(np.conj(Xb[:,:,None,:]), Inv_Sigma_x)
        xSx = np.real(np.matmul(xS, Xb[:,:,:,None]))
        # xS = np.matmul(Xb[:,:,:,None], np.conj(Xb[:,:,None,:]))
        # xSx1 = np.trace(np.matmul(xS, Inv_Sigma_x), axis1=, axis2=-1))
        # xSx1 = np.trace(np.real(np.matmul(xS, Inv_Sigma_x)), axis1=2, axis2=3)
        # print(xSx1.shape)
        # xSx1 = np.zeros((F,N))
        # for i in range(I):
        #     for ii in range(I):
        #         print(i, ii)
        #         if i == ii:
        #             xSx1 += np.real(Inv_Sigma_x[:,:,i,ii] * np.abs(Xb[:,:,ii])**2)
        #         else:
        #             xSx1 += np.real(Inv_Sigma_x[:,:,i,ii] * np.conj(Xb[:,:,i]) * Xb[:,:,ii])
        # log_like1 = - np.sum( xSx1 + np.log(Det_Sigma_x * np.pi)) / (N * F)
        log_like = - np.sum( np.squeeze(xSx) + np.log(Det_Sigma_x * np.pi)) / (N * F)

        if iter > 1:
            log_like_diff = log_like - log_like_arr[iter - 1]
            print('Log-likelihood: {}   Log-likelihood improvement: {}'.format(log_like, log_like_diff))
        else:
            print('Log-likelihood:', log_like)
        log_like_arr[iter] = log_like

        for j in range(J):
            # compute S-Wiener gain
            Gs[:,:,j,:] = ( np.matmul(np.conj(A[:,None,None,:,j]), Inv_Sigma_x).squeeze()\
                             * sigma_ss[:,:,j,None])

            # compute Gs_x
            Gs_x[:,:,j] = np.einsum('fti,fti->ft', Gs[:,:,j,:], Xb)

            # compute average Rxs
            for i in range(I):
                bar_Rxs[:,i,j] = np.mean(Xb[:,:,i] * np.conj(Gs_x[:,:,j]), axis=1)

        for j1 in range(J):
            # compute average Rss
            for j2 in range(J):
                GsA = (np.matmul(Gs[:,:,j1,None,:], A[:,None,:,j2,None])).squeeze()
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
            sigma_cc_k = np.dot(W[:,k,np.newaxis], H[np.newaxis,k,:])

            # compute C-Wiener gain
            Gc_k = ( np.matmul(np.conj(bar_A[:,None,None,:,k]), Inv_Sigma_x).squeeze()\
                 * sigma_cc_k[:,:,None])

            # compute Gc_x
            Gc_x_k = np.einsum('fni,fni->fn', Gc_k, Xb)


            # compute components sufficient natural statistics
            # IT IS IMPORTANT TO TAKE A REAL PART !!!!
            Vc[:,:,k] = np.abs(Gc_x_k) ** 2 + sigma_cc_k \
                     - np.real(
                          np.matmul(Gc_k,bar_A[:,:,k,None]).squeeze()
                       ) * sigma_cc_k

        # M-step: re-estimate
        print('   M-step')

        # re-estimate A
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
            W[:, k] = np.sum(Vc[:,:,k] / np.outer(np.ones(F), H[k, :]), axis=1) / N
            H[k, :] = np.sum(Vc[:,:,k] / np.outer(W[:,k], np.ones(N)), axis=0) / F

        for j in range(J):
            nonzero_f_ind = np.where(A[:,0,j] != 0)[0]
            sign = A[nonzero_f_ind,0,j] / np.abs(A[nonzero_f_ind,0,j])
            for i in range(I):
                A[nonzero_f_ind,I-i-1,j] = A[nonzero_f_ind,I-i-1,j] / sign

            A_scale = np.zeros(F)
            for i in range(I):
                A_scale += np.abs(A[:,i,j]) ** 2
            A[:,:,j] = A[:,:,j] / np.sqrt(A_scale[:,np.newaxis])
            W[:,source_NMF_ind[j]] = W[:, source_NMF_ind[j]] * A_scale[:,np.newaxis]

        # Normalisation of W components
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


def multinmf_conv_em_wrapper(x, partial_rirs, n_latent_var, n_iter = 500, verbose = False):
    '''
    A wrapper around multichannel nmf using EM updates to use with pyroormacoustcs.
    Performs STFT and ensures all signals are the correct shape.

    Parameters
    ----------
    x: ndarray
        (n_samples x n_channel) array of time domain samples
    partial_rirs: ndarray
        (n_channel x n_src x n_bins) array of partial TF
    n_latent_var: int
        number of latent variables in the NMF
    '''

    n_channel = x.shape[1]
    n_src = partial_rirs.shape[1]
    stft_win_len = 2 * (partial_rirs.shape[2] - 1)

    # STFT
    window = np.sqrt(pra.cosine(stft_win_len))  # use sqrt because of synthesis
    # X is (n_channel, n_frame, n_bin)
    X = np.array(
            [pra.stft(x[:,ch], stft_win_len, stft_win_len // 2, win=window, transform=np.fft.rfft) for ch in range(n_channel)]
            )
    # move axes to match Ozerov's order (n_bin, n_frame, n_channel)
    X = np.moveaxis(X, [0,1,2], [2,1,0])
    n_bin = X.shape[0]
    n_frame = X.shape[1]

    # Random initialization of multichannel NMF parameters
    K = n_latent_var * n_src
    source_NMF_ind = []
    for j in range(n_src):
        source_NMF_ind = np.reshape(np.arange(n_latent_var * n_src, dtype=np.int), (n_src,-1))

    mix_psd = 0.5 * (np.mean(np.abs(X[:,:,0])**2 + np.abs(X[:,:,1])**2, axis=1))
    A_init = (0.5 *
                ( 1.9 * np.abs(random.randn(n_bin, n_channel, n_src))       \
                + 0.1 * np.ones((n_bin, n_channel, n_src))                  \
                ) * np.sign( random.randn(n_bin, n_channel, n_src)          \
                            + 1j * random.randn(n_bin, n_channel, n_src))  \
            )
    # W is intialized so that its enegy follows mixture PSD
    W_init = 0.5 * (
            ( np.abs(np.random.randn(n_bin,K)) + np.ones((n_bin,K)) )
            * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
            )
    H_init = 0.5 * ( np.abs(np.random.randn(K,n_frame)) + np.ones((K,n_frame)) )
    Sigma_b_init = mix_psd / 100

    # squared mag partial rirs (n_bin, n_channel, n_src)
    Q_init = np.moveaxis(np.abs(partial_rirs)**2, [2], [0])

    W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr = \
        multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind, iter_num=500)

    Ae_EM = np.moveaxis(Ae_EM, [0], [2])

    # Computation of the spatial source images
    print('Computation of the spatial source images\n')
    Ie_EM = np.zeros((n_bin,n_fram,n_src,nchan), dtype=np.complex)
    for j in range(n_src):
        for f in range(n_bin):
            Ie_EM[f,:,j,:] = np.outer(Se_EM[f,:,j], Ae_EM[:,j,f])

    sep_sources = []
    # Inverse STFT
    for j in range(n_src):
        # channel-wise istft with synthesis window
        ie_MU = []
        for ch in range(n_channel):
            ie_MU.append(
                    pra.istft(Im[:,:,j,ch].T, stft_win_len, stft_win_len // 2, win=window, transform=np.fft.irfft)
                    )

        sep_sources.append(np.array(ie_MU).T)

    return np.array(sep_sources)
