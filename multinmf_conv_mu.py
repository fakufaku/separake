import numpy as np

def multinmf_conv_mu(V, W, H, Q, part, n_iter=500, fix_Q=False, fix_W=False, fix_H=False, verbose=False):
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

    # Definitions
    V_ap = np.zeros((F,N,n_c))
    cost = np.zeros(n_iter)

    # a bunch of helper functions
    def update_approximation(V_ap, Q, P):
        ''' Recompute the approximation '''
        V_ap += Q[:,np.newaxis,:] * P[:,:,np.newaxis]

    def compute_cost(V, V_ap):
        ''' Compute divergence cost function '''
        return np.sum(V / V_ap - np.log(V / V_ap)) - np.prod(V.shape)

    # Compute app. variance structure V_ap
    for j in range(n_s):
        P_j = np.dot(W[:,part[j]], H[part[j],:])
        update_approximation(V_ap, Q[:,:,j], P_j)

    # initial cost
    cost[0] = compute_cost(V, V_ap)

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
            cost[iter] = compute_cost(V, V_ap)
            print('MU update: iteration', iter, 'of', n_iter, ', cost =', cost[iter])

    return W, H, Q, cost

