
import numpy as np

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

    if I != 2:
        raise ValueError('Multi_NMF_EM_conv: number of channels must be 2')

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
    Sigma_x = np.zeros((F, N, 2, 2), dtype=np.complex)
    Inv_Sigma_x = np.zeros((F, N, 2, 2), dtype=np.complex)
    Gs = np.zeros((F, N, J, 2), dtype=np.complex)
    Gs_x = np.zeros((F, N, J), dtype=np.complex)
    bar_Rxs = np.zeros((F, 2, J), dtype=np.complex)
    bar_Rss = np.zeros((F, J, J))
    bar_Rxx = np.zeros((F, 2, 2), dtype=np.complex)
    bar_A = np.zeros((F, 2, K), dtype=np.complex)
    Vc = np.zeros((F, N, K))
    log_like_arr = np.zeros((iter_num))

    # initialize simulated annealing variances (if necessary)
    if SimAnneal_flag > 0:
        Sigma_b_anneal = np.zeros((F, iter_num))
        for iter in range(iter_num):
            Sigma_b_anneal[:, iter] = ((np.sqrt(Sigma_b0) * (iter_num - iter) + np.ones(F) * np.sqrt(final_ann_noise_var) * iter) / iter_num) ** 2

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
            for k in range(len(source_NMF_ind[j])):
                sigma_ss[:,:,j] += np.dot(W[:,source_NMF_ind[j][k],np.newaxis], H[np.newaxis,source_NMF_ind[j][k],:])

        if SimAnneal_flag:
            Sigma_b = Sigma_b_anneal[:,iter]

            if SimAnneal_flag == 2:  # with noise injection
                Noise = np.random.randn(F, N, 2) + 1j * np.random.randn(F, N, 2)  # complex noise
                Noise[:,:,0] *= np.outer(np.sqrt(Sigma_b / 2), np.ones(N))
                Noise[:,:,1] *= np.outer(np.sqrt(Sigma_b / 2), np.ones(N))

                Xb = X + Noise

        # compute the Sigma_x matrix
        Sigma_x[:,:,0,0] = np.outer(Sigma_b, O)
        Sigma_x[:,:,0,1] = 0
        Sigma_x[:,:,1,0] = 0
        Sigma_x[:,:,1,1] = np.outer(Sigma_b, O)
        for j in range(J):
            Sigma_x[:,:,0,0] += np.outer(np.abs(A[:,0,j]) ** 2, O) * sigma_ss[:,:,j]
            cc = A[:,0,j] * np.conj(A[:,1,j])
            Sigma_x[:,:,0,1] += np.outer(cc[:], O) * sigma_ss[:, :, j]
            Sigma_x[:,:,1,0] = np.conj(Sigma_x[:,:,0,1])
            Sigma_x[:,:,1,1] += np.outer(np.abs(A[:,1,j]) ** 2, O) * sigma_ss[:,:,j]

        # compute the inverse of Sigma_x matrix
        Det_Sigma_x = Sigma_x[:,:,0,0] * Sigma_x[:,:,1,1] - np.abs(Sigma_x[:, :,0,1]) ** 2
        Inv_Sigma_x[:,:,0,0] = Sigma_x[:,:,0,0] / Det_Sigma_x
        Inv_Sigma_x[:,:,0,1] = -Sigma_x[:,:,0,1] / Det_Sigma_x
        Inv_Sigma_x[:,:,1,0] = np.conj(Inv_Sigma_x[:,:,0,1])
        Inv_Sigma_x[:,:,1,1] = Sigma_x[:,:,0,0] / Det_Sigma_x

        # compute log-likelihood
        log_like = - np.sum( 
                        np.log(Det_Sigma_x * np.pi)
                        + Inv_Sigma_x[:,:,0,0] * np.abs(Xb[:,:,0]) ** 2
                        + Inv_Sigma_x[:,:,1,1] * np.abs(Xb[:,:,1]) ** 2
                        + 2. * np.real(Inv_Sigma_x[:,:,0,1] * np.conj(Xb[:,:,0]) * Xb[:,:,1])
                        ) / (N * F)

        if iter > 1:
            log_like_diff = log_like - log_like_arr[iter - 1]
            print('Log-likelihood: {}   Log-likelihood improvement: {}'.format(log_like, log_like_diff))
        else:
            print('Log-likelihood:', log_like)
        log_like_arr[iter] = log_like

        for j in range(J):
            # compute S-Wiener gain
            Gs[:,:,j,0] = np.outer(np.conj(A[:,0,j]), O) * Inv_Sigma_x[:,:,0,0] + \
                          np.outer(np.conj(A[:,1,j]), O) * Inv_Sigma_x[:,:,1,0] * sigma_ss[:,:,j]
            Gs[:,:,j,1] = np.outer(np.conj(A[:,0,j]), O) * Inv_Sigma_x[:,:,0,1] + \
                          np.outer(np.conj(A[:,1,j]), O) * Inv_Sigma_x[:,:,1,1] * sigma_ss[:,:,j]

            # compute Gs_x
            Gs_x[:,:,j] = Gs[:,:,j,0] * Xb[:,:,0] + Gs[:,:,j,1] * Xb[:,:,1]

            # compute average Rxs
            bar_Rxs[:,0,j] = np.mean(Xb[:,:,0] * np.conj(Gs_x[:,:,j]), axis=1)
            bar_Rxs[:,1,j] = np.mean(Xb[:,:,1] * np.conj(Gs_x[:,:,j]), axis=1)

        for j1 in range(J):
            # compute average Rss
            for j2 in range(J):
                bar_Rss[:,j1,j2] = np.mean(
                        Gs_x[:,:,j1] * np.conj(Gs_x[:,:,j2]) 
                        - ( Gs[:,:,j1,0] * np.outer(A[:,0,j2], O) + Gs[:,:,j1,1] * np.outer(A[:,1,j2], O)) * sigma_ss[:,:,j2], 
                        axis=1)
            bar_Rss[:,j1,j1] = bar_Rss[:,j1,j1] + np.mean(sigma_ss[:,:,j1], axis=1)

        # compute average Rxx
        bar_Rxx[:,0,0] = np.mean(abs(Xb[:,:,0]) ** 2, axis=1)
        bar_Rxx[:,1,1] = np.mean(abs(Xb[:,:,1]) ** 2, axis=1)
        bar_Rxx[:,0,1] = np.mean(Xb[:,:,0] * np.conj(Xb[:,:,1]), axis=1)
        bar_Rxx[:,1,0] = np.conj(bar_Rxx[:, 0, 1])

        # TO ASSURE that Rss = Rss'
        for f in range(F):
            bar_Rss[f,:,:] = (np.squeeze(bar_Rss[f,:,:]) + np.squeeze(bar_Rss[f,:,:]).T) / 2

        # compute extended mixing matrix A
        for j in range(J):
            for k in source_NMF_ind[j]:
                bar_A[:,:,k] = A[:,:,j]

        for k in range(K):
            # compute a priori component variances
            sigma_cc_k = np.dot(W[:,k,np.newaxis], H[np.newaxis,k,:])

            # compute C-Wiener gain
            Gc_k_1 = ( np.outer(np.conj(bar_A[:,0,k]), O) * Inv_Sigma_x[:,:,0,0] + np.outer(np.conj(bar_A[:,1,k]), O) * Inv_Sigma_x[:,:,1,0]) * sigma_cc_k
            Gc_k_2 = ( np.outer(np.conj(bar_A[:,0,k]), O) * Inv_Sigma_x[:,:,0,1] + np.outer(np.conj(bar_A[:,1,k]), O) * Inv_Sigma_x[:,:,1,1]) * sigma_cc_k

            # compute Gc_x
            Gc_x_k = Gc_k_1 * Xb[:, :, 0] + Gc_k_2 * Xb[:, :, 1]

            # compute components sufficient natural statistics
            # IT IS IMPORTANT TO TAKE A REAL PART !!!!
            Vc[:, :, k] = np.abs(Gc_x_k) ** 2 + sigma_cc_k - np.real(Gc_k_1 * np.outer(bar_A[:,0,k], O) + Gc_k_2 * np.outer(bar_A[:,1,k], O)) * sigma_cc_k

        # M-step: re-estimate parameters
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

        # Normalization
        for j in range(J):
            nonzero_f_ind = np.where(A[:,0,j] != 0)[0]
            A[nonzero_f_ind,1,j] = A[nonzero_f_ind,1,j] / np.sign(A[nonzero_f_ind,0,j])
            A[nonzero_f_ind,0,j] = A[nonzero_f_ind,0,j] / np.sign(A[nonzero_f_ind,0,j])

            A_scale = np.abs(A[:,0,j]) ** 2 + np.abs(A[:,1,j]) ** 2
            A[:,:,j] = A[:,:,j] / np.outer(np.sqrt(A_scale), np.ones(2))
            W[:,source_NMF_ind[j]] = W[:, source_NMF_ind[j]] * np.outer(A_scale, np.ones(len(source_NMF_ind[j])))

        # Normalisation of W components
        w = np.sum(W, axis=0)
        d = np.diag(np.ones(K) / w)
        W = np.dot(W, d)
        # Energy transfer to H
        d = np.diag(w)
        H = np.dot(d, H)

    # source estimates
    S = Gs_x

    # return parameters estimated on previous iteration, since they were used
    # for source estimates computation
    W = W_prev
    H = H_prev
    A = A_prev
    Sigma_b = Sigma_b_prev

    return W, H, A, Sigma_b, S, log_like_arr
