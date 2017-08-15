
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
    J = source_NMF_ind.shape[0]

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

    O = np.ones((1, N))

    sigma_ss = np.zeros((F, N, J))
    Sigma_x = np.zeros((F, N, 2, 2))
    Inv_Sigma_x = np.zeros((F, N, 2, 2))
    Gs = np.zeros((F, N, J, 2))
    Gs_x = np.zeros((F, N, J))
    bar_Rxs = np.zeros((F, 2, J))
    bar_Rss = np.zeros((F, J, J))
    bar_Rxx = np.zeros((F, 2, 2))
    bar_A = np.zeros((F, 2, K))
    Vc = np.zeros((F, N, K))
    log_like_arr = np.zeros((1, iter_num))

    # initialize simulated annealing variances (if necessary)
    if SimAnneal_flag > 0:
        Sigma_b_anneal = np.zeros((F, iter_num))
        for iter in range(iter_num):
            Sigma_b_anneal[:, iter] = ((np.sqrt(Sigma_b0) * (iter_num - iter) + np.ones(F, 1) * np.sqrt(final_ann_noise_var) * iter) / iter_num) ** 2

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
                sigma_ss[:,:,j] = sigma_ss[:,:,j] + np.dot(W[:,source_NMF_ind[j][k],np.newaxis], H[np.newaxis,source_NMF_ind[j][k],:])

        if SimAnneal_flag:
            Sigma_b = Sigma_b_anneal[:,iter,np.newaxis]

            if SimAnneal_flag == 2:  # with noise injection
                Noise = np.random.randn(F, N, 2) + 1j * np.random.randn(F, N, 2)  # complex noise
                Noise[:,:,1] *= np.dot(np.sqrt(Sigma_b / 2) * np.ones((1, N)))))
                Noise[:,:,2] *= np.dot(np.sqrt(Sigma_b / 2) * np.ones((1, N)))))

                Xb = X + Noise

        # compute the Sigma_x matrix
        Sigma_x[:,:,1,1] = np.dot(Sigma_b, O)
        Sigma_x[:,:,1,2] = 0
        Sigma_x[:,:,2,1] = 0
        Sigma_x[:,:,2,2] = np.dot(Sigma_b, O)
        for j in range(J):
            Sigma_x[:, :, 1, 1] += np.dot(np.abs(A[:,1,j]) ** 2, O) * sigma_ss[:,:,j]
            Sigma_x[:, :, 1, 2] += np.dot((A[:,1,j] * np.conj(A[:,2,j])), O) * sigma_ss[:, :, j]
            Sigma_x[:, :, 2, 1] = np.conj(Sigma_x[:,:,1,2])
            Sigma_x[:, :, 2, 2] += np.dot(np.abs(A[:,2,j]) ** 2, O) * sigma_ss[:,:,j]

        # compute the inverse of Sigma_x matrix
        Det_Sigma_x = Sigma_x[:,:,1,1] * Sigma_x[:, :, 2, 2] - np.abs(Sigma_x[:, :, 1, 2]) ** 2
        Inv_Sigma_x[:,:,1,1] = Sigma_x[:,:,2,2] / Det_Sigma_x
        Inv_Sigma_x[:,:,1,2] = -Sigma_x[:,:,1,2] / Det_Sigma_x
        Inv_Sigma_x[:,:,2,1] = np.conj(Inv_Sigma_x[:,:,1,2])
        Inv_Sigma_x[:,:,2,2] = Sigma_x[:,:,1,1] / Det_Sigma_x

        # compute log-likelihood
        log_like = - np.sum( 
                        np.log(Det_Sigma_x * np.pi)
                        + Inv_Sigma_x[:,:,1,1] * np.abs(Xb[:,:,1]) ** 2
                        + Inv_Sigma_x[:,:,2,2] * np.abs(Xb[:,:,2]) ** 2
                        + 2. * np.real(Inv_Sigma_x[:,:,1,2] * np.conj(Xb[:,:,1]) * Xb[:,:,2])
                        ) / (N * F)

        if iter > 1:
            log_like_diff = log_like - log_like_arr[iter - 1]
            print('Log-likelihood: {}   Log-likelihood improvement: {}'.format(log_like, log_like_diff))
        else:
            print('Log-likelihood:', log_like)
        log_like_arr[iter] = log_like

        for j in range(J):
            # compute S-Wiener gain
            Gs[:, :, j, 1] = multiply((multiply((dot(conj(A[:,1,j]), O)), Inv_Sigma_x[:,:,1,1]) + multiply(
                (dot(conj(A[:,2,j]), O)), Inv_Sigma_x[:,:,2,1])), sigma_ss[:, :, j])
            Gs[:, :, j, 2] = multiply((multiply((dot(conj(A[:,1,j]), O)), Inv_Sigma_x[:,:,1,2]) + multiply(
                (dot(conj(A[:,2,j]), O)), Inv_Sigma_x[:,:,2, 2])), sigma_ss[:, :, j])

            # compute Gs_x
            Gs_x[:,:,j] = Gs[:,:,j,1] * Xb[:,:,1] + Gs[:,:,j,2] * Xb[:,:,2]

            # compute average Rxs
            bar_Rxs[:,1,j] = np.mean(Xb[:,:,1] * np.conj(Gs_x[:,:,j]), axis=1)
            bar_Rxs[:,2,j] = np.mean(Xb[:,:,2] * np.conj(Gs_x[:,:,j]), axis=1)

        for j1 in range(J):
            # compute average Rss
            for j2 in range(J):
                bar_Rss[:,j1,j2] = np.mean(
                        Gs_x[:,:,j1] * np.conj(Gs_x[:,:,j2]) 
                        - ( Gs[:,:,j1,1] * np.dot(A[:,1,j2], O) + Gs[:,:,j1, 2] * np.dot(A[:,2,j2], O)) * sigma_ss[:,:,j2], 
                        axis=1)
            bar_Rss[:,j1,j1] = bar_Rss[:,j1,j1] + np.mean(sigma_ss[:,:,j1], axis=1)

        # compute average Rxx
        bar_Rxx[:,1,1] = np.mean(abs(Xb[:,:,1]) ** 2, axis=1)
        bar_Rxx[:,2,2] = np.mean(abs(Xb[:,:,2]) ** 2, axis=1)
        bar_Rxx[:,1,2] = np.mean(Xb[:,:,1] * conj(Xb[:,:,2]), axis=1)
        bar_Rxx[:,2,1] = conj(bar_Rxx[:, 1, 2])

        # TO ASSURE that Rss = Rss'
        for f in range(F):
            bar_Rss[f,:,:] = np.squeeze(bar_Rss[f,:,:]) + np.squeeze(bar_Rss[f,:,:]).T) / 2

        # compute extended mixing matrix A
        for j in range(J):
            for k in source_NMF_ind[j]:
                bar_A[:,:,k] = A[:,:,j]

        for k in range(K):
            # compute a priori component variances
            sigma_cc_k = np.dot(W[:,k,np.newaxis], H[np.newaxis,k,:])

            # compute C-Wiener gain
            Gc_k_1 = ( np.dot(np.conj(bar_A[:,1,k]), O) * Inv_Sigma_x[:,:,1,1] + np.dot(np.conj(bar_A[:,2,k]), O) * Inv_Sigma_x[:,:,2,1]) * sigma_cc_k
            Gc_k_2 = ( np.dot(np.conj(bar_A[:,1,k]), O) * Inv_Sigma_x[:,:,1,2] + np.dot(np.conj(bar_A[:,2,k]), O) * Inv_Sigma_x[:,:,2,2]) * sigma_cc_k

            # compute Gc_x
            Gc_x_k = Gc_k_1 * Xb[:, :, 1] + Gc_k_2 * Xb[:, :, 2]

            # compute components sufficient natural statistics
            # IT IS IMPORTANT TO TAKE A REAL PART !!!!
            Vc[:, :, k] = np.abs(Gc_x_k) ** 2 + sigma_cc_k - np.real(Gc_k_1 * np.dot( bar_A[:,1,k], O) + Gc_k_2 * np.dot(bar_A[:,2,k], O)) * sigma_cc_k

        # M-step: re-estimate parameters
        print('   M-step')

        # re-estimate A
        for f in range(F):
            A[f,:,:] = np.dot(np.squeeze(np.mean(bar_Rxs[f,:,:], axis=0)), np.linalg.inv(np.squeeze(np.mean(bar_Rss[f,:,:], axis=0))))

        # re-estimate noise variances (if necessary)
        if Sigma_b_Upd_flag:
            for f in range(F):
                Sigma_b[f] = 0.5 * np.real(
                        np.trace(
                            np.squeeze(bar_Rxx[f, :, :]) - 
                            np.dot(np.squeeze(A[f,:,:]), np.conj(np.squeeze(bar_Rxs[f, :, :]).T)) - 
                            np.dot(np.squeeze(bar_Rxs[f,:,:]), np.conj(squeeze(A[f, :, :]).T)) + 
                            np.dot(dot(squeeze(A[f, :, :]), squeeze(bar_Rss[f, :, :])), np.conj(squeeze(A[f, :, :]).T))
                            )
                        )
        # re-estimate W, and then H
        for k in range(K):
            W[:, k] = np.sum(Vc[:,:,k] / np.outer(np.ones(F), H[k, :]), axis=1) / N
            H[k, :] = np.sum(Vc[:,:,k] / np.outer(W[:,k], np.ones(N)), axis=0) / F

        # Normalization
        for j in range(J):
            nonzero_f_ind = np.where(A[:,1,j] != 0)[0]
            A[nonzero_f_ind,2,j] = A[nonzero_f_ind,2,j] / np.sign(A[nonzero_f_ind,1,j])
            A[nonzero_f_ind,1,j] = A[nonzero_f_ind,1,j] / np.sign(A[nonzero_f_ind,1,j])
            A_scale = np.abs(A[:,1,j]) ** 2 + np.abs(A[:,2,j]) ** 2
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