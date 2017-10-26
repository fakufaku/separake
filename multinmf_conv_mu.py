import numpy as np
import pyroomacoustics as pra

from multinmf_recons_im import multinmf_recons_im

def multinmf_conv_mu(V, W, H, Q, part, n_iter=500, fix_Q=False, fix_W=False, fix_H=False, H_l1_reg=0., smooth_reg=0., smoothing_matrix=None, verbose=False):
    '''
    Multichannel NMF minimizing Itakura-Saito divergence through multiplicative updates

    [Q, W, H, cost] = multinmf_conv_mu(V, n_iter, Q, W, H, part, switch_Q, switch_W, switch_H)

    Sizes
    * F: number of frequency bins
    * N: number of frames
    * K: number of latent variables in the non-negative decomposition
    * n_c: number of channels
    * n_s: number of sources

    Parameters
    ----------
    V:
        positive matrix data       (F x N x n_c)
    n_iter: int
        number of iterations
    W: nd-array
        basis                 (F x K)
    H: nd-array
        activation coef.      (K x N)
    Q: nd-array
        squared magnitude channels filters       (F x n_c x n_s)
    part: list
        W and H are shared for all sources, this is a list of size n_s with
        i-th element being the list of indices in W and H corresponding to the
        i-th source
    fix_Q: bool, optional
        When True, matrix Q is kept fixed (default False)
    fix_W: bool, optional
        When True, matrix W is kept fixed (default False)
    fix_H: bool, optional
        When True, matrix H is kept fixed (default False)
    H_l1_reg: float
        The weight of the l1 (sparsity) regularizer
    verbose: bool, optional
        Show more information


    Returns
    -------
      - Estimated Q, W and H
      - Cost through iterations betw. data power and fitted variance.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Copyright 2017 Robin Scheibler robin, ported to python
    Copyright 2010 Cedric Fevotte
    (cedric.fevotte -at- telecom-paristech.fr)

    This software is distributed under the terms of the GNU Public License
    version 3 (http://www.gnu.org/licenses/gpl.txt)

    If you use this code please cite this paper

    A. Ozerov and C. Fevotte,
    "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
    IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
    of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
    Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Stable version (11/09/08); scales fixed properly.
    This version updates V_ap after each parameter update, leading to faster
    convergence
    '''

    F, N, n_c = V.shape
    n_s = Q.shape[2]

    reg_norm = (n_c * F) / W.shape[1]

    # Definitions
    V_ap = np.zeros((F,N,n_c))
    cost = np.zeros(n_iter)

    # a bunch of helper functions
    def update_approximation(V_ap, Q, P):
        ''' Recompute the approximation '''
        V_ap += Q[:,np.newaxis,:] * P[:,:,np.newaxis]

    def compute_cost(V, V_ap, H, l1_reg):
        ''' Compute divergence cost function '''
        return np.sum(V / V_ap - np.log(V / V_ap)) - np.prod(V.shape) + reg_norm * l1_reg * np.sum(H)

    # Compute app. variance structure V_ap
    for j in range(n_s):
        P_j = np.dot(W[:,part[j]], H[part[j],:])
        update_approximation(V_ap, Q[:,:,j], P_j)

    # initial cost
    cost[0] = compute_cost(V, V_ap, H, H_l1_reg)

    # start the iteration
    for iter in range(1,n_iter):

        if not fix_Q:
            ''' Update Q '''
            for j in range(n_s):
                P_j = np.dot(W[:,part[j]], H[part[j],:])
                Q_old = np.copy(Q[:,:,j])

                for i in range(n_c):
                    Q[:,i,j] *= ( np.dot(V[:,:,i] * P_j / V_ap[:,:,i]**2, np.ones(N))
                            / np.dot(P_j / V_ap[:,:,i], np.ones(N)) )

                update_approximation(V_ap, Q[:,:,j] - Q_old, P_j)

        if not fix_W:
            ''' Update W '''
            for j in range(n_s):
                Kj = len(part[j])

                Wnum = np.zeros((F,Kj))
                Wden = np.zeros((F,Kj))
                one_over_V_ap = 1 / V_ap
                for i in range(n_c):
                    Wnum += Q[:,i,j,np.newaxis] * np.dot(V[:,:,i] * one_over_V_ap[:,:,i]**2, H[part[j],:].T)
                    Wden += Q[:,i,j,np.newaxis] * np.dot(one_over_V_ap[:,:,i], H[part[j],:].T)

                W_old = np.copy(W[:,part[j]])
                W[:,part[j]] *= (Wnum / Wden)

                P_j = np.dot(W[:,part[j]] - W_old, H[part[j],:])
                update_approximation(V_ap, Q[:,:,j], P_j)

        if not fix_H:
            ''' Update H '''
            for j in range(n_s):
                Kj = len(part[j])

                Hnum = np.zeros((Kj,N))
                Hden = np.zeros((Kj,N))
                one_over_V_ap = 1 / V_ap
                for i in range(n_c):
                    QW = Q[:,i,j,np.newaxis] * W[:,part[j]]
                    Hnum += np.dot(QW.T, V[:,:,i] * one_over_V_ap[:,:,i]**2)
                    Hden += np.dot(QW.T, one_over_V_ap[:,:,i])

                # regularize for sparsity if required
                if H_l1_reg != 0.:
                    Hden += reg_norm * H_l1_reg

                H_old = np.copy(H[part[j],:])
                H[part[j],:] *= (Hnum / Hden)

                P_j = np.dot(W[:,part[j]], H[part[j],:] - H_old)
                update_approximation(V_ap, Q[:,:,j], P_j)

        ### Normalize ###

        ## Scale Q / W ##
        if (not fix_Q) and (not fix_W):
            # normalize so that the squared magnitude filters sum to one
            # over the channels
            scale = np.sum(Q, axis=1)
            Q /= scale[:,np.newaxis,:]
            # the W matrices need to be normalized source-wise
            for j in range(n_s):
                W[:,part[j]] *= scale[:,j,np.newaxis]

        ## Scale W / H ##
        if (not fix_W) and (not fix_H):
            # normalize so that the columns of W sum to unity
            scale = np.sum(W, axis=0)
            W /= scale[np.newaxis,:]
            H *= scale[:,np.newaxis]

        if verbose and (iter % 25) == 0:
            cost[iter] = compute_cost(V, V_ap, H, H_l1_reg)
            print('MU update: iteration', iter, 'of', n_iter, ', cost =', cost[iter])

    return W, H, Q, cost

def multinmf_conv_mu_wrapper(x, n_src, n_latent_var, stft_win_len, partial_rirs=None, W_dict=None, n_iter=500, l1_reg=0., random_seed=0, verbose=False):
    '''
    A wrapper around multichannel nmf using MU updates to use with pyroormacoustcs.
    Performs STFT and ensures all signals are the correct shape.

    Parameters
    ----------
    x: ndarray
        (n_samples x n_channel) array of time domain samples
    n_src: int
        The number of sources
    n_latent_var: int
        The number of latent variables in the NMF
    stft_win_len:
        The length of the STFT window
    partial_rirs: array_like, optional
        (n_channel x n_src x n_bins) array of partial TF. If provided, Q is not optimized.
    W_dict: array_like, optional
        A dictionary of atoms that can be used in the NMF. If provided, W is not optimized.
    n_iter: int, optional
        The number of iterations of NMF (default 500)
    l1_reg: float, optional
        The weight of the l1 regularization term for the activations (default 0., not regularized)
    random_seed: unsigned int, optional
        The seed to provide to the RNG prior to initialization of NMF parameters. This allows to use
        repeatable initialization.
    verbose: bool, optional
        When true, prints convergence info of NMF (default False)
    '''

    n_channel = x.shape[1]

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

    # Squared magnitude and unit energy per bin
    V = np.abs(X)**2
    V /= np.mean(V)

    # Random initialization of multichannel NMF parameters
    np.random.seed(random_seed)

    K = n_latent_var * n_src
    source_NMF_ind = []
    for j in range(n_src):
        source_NMF_ind = np.reshape(np.arange(n_latent_var * n_src, dtype=np.int), (n_src,-1))

    mix_psd = np.mean(V, axis=(1,2))
    # W is intialized so that its enegy follows mixture PSD
    if W_dict is None:
        W_init = 0.5 * (
                ( np.abs(np.random.randn(n_bin,K)) + np.ones((n_bin,K)) )
                * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
                )
        fix_W = False
    else:
        if W_dict.shape[1] == n_latent_var:
            W_init = np.tile(W_dict, n_src)
        elif W_dict.shape[1] == n_src * n_latent_var:
            W_init = W_dict
        else:
            raise ValueError('Mismatch between dictionary size and latent variables')
        fix_W = True

    W_init /= np.sum(W_init, axis=0)[None,:]

    # follow average activations
    mix_act = np.mean(V, axis=(0,2))
    H_init = 0.5 * ( np.abs(np.random.randn(K,n_frame)) + np.ones((K,n_frame)) ) * mix_act[np.newaxis,:]

    if partial_rirs is not None:
        # squared mag partial rirs (n_bin, n_channel, n_src)
        Q_init = np.moveaxis(np.abs(partial_rirs)**2, [2], [0])
        Q_init /= np.max(Q_init, axis=0)[None,:,:]
        fix_Q = True
    else:
        # random initialization
        Q_shape = (n_bin, n_channel, n_src)
        Q_init = (0.5 * (1.9 * np.abs(np.random.randn(*Q_shape)) + 0.1 * np.ones(Q_shape))) ** 2
        fix_Q = False

    # RUN NMF
    W_MU, H_MU, Q_MU, cost = \
        multinmf_conv_mu(
                np.abs(X)**2, W_init, H_init, Q_init, source_NMF_ind, 
                n_iter=n_iter, fix_Q=fix_Q, fix_W=fix_W, 
                H_l1_reg=l1_reg, 
                verbose=verbose)

    # Computation of the spatial source images
    Im = multinmf_recons_im(X, W_MU, H_MU, Q_MU, source_NMF_ind)

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
