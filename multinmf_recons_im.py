import numpy as np

def multinmf_recons_im(X, W, H, Q, part):
    '''
    Reconstruction of sources from their non-negative factorization
    with non-negative channel weights.

    Sizes
    * F: number of frequency bins
    * N: number of frames
    * K: number of latent variables in the non-negative decomposition
    * n_c: number of channels
    * n_s: number of sources
   
    Parameters
    ----------
    X:
        STFT of multichannel mixture       (F x N x n_c)
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

    '''

    F,N,n_c = X.shape  # n_bins, n_frames, n_channels
    n_s = Q.shape[2]   # n_sources

    P = np.zeros((F,N,n_s), dtype=np.float)
    Im = np.zeros((F,N,n_s,n_c), dtype=np.complex)

    # compute the separated spectrogram of all sources
    # from their factorized parts
    for j in range(n_s):
        P[:,:,j] = np.dot(W[:,part[j]], H[part[j],:])

    # Re-synthetize the 
    for i in range(n_c):

        # recompute the approximated non-negative spectrogram at microphone i
        P_at_mic = np.zeros(P.shape, dtype=np.float)
        for j in range(n_s):
            P_at_mic[:,:,j] = Q[:,i,j,np.newaxis] * P[:,:,j]

        # mix all sources
        P_at_mic_all_sources = np.sum(P_at_mic, axis=2)

        # Procedure:
        # 1) Divide original spectrogram by the cumulative non-negative approximation
        # 2) Multiply by the separated source magnitude
        for j in range(n_s):
            Im[:,:,j,i] = P_at_mic[:,:,j] / P_at_mic_all_sources * X[:,:,i]

    return Im


