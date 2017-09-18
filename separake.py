
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

from multinmf_conv_mu import multinmf_conv_mu
from multinmf_recons_im import multinmf_recons_im

# get the speed of sound from pyroomacoustics
c = pra.constants.get('c')

def multinmf_conv_mu_wrapper(x, partial_rirs, n_latent_var, n_iter=500):
    '''
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
    print('Random initialization of multichannel NMF parameters')
    K = n_latent_var * n_src
    source_NMF_ind = []
    for j in range(n_src):
        source_NMF_ind = np.reshape(np.arange(n_latent_var * n_src, dtype=np.int), (n_src,-1))

    mix_psd = 0.5 * (np.mean(np.abs(X[:,:,0])**2 + np.abs(X[:,:,1])**2, axis=1))
    # W is intialized so that its enegy follows mixture PSD
    W_init = 0.5 * (
            ( np.abs(np.random.randn(n_bin,K)) + np.ones((n_bin,K)) ) 
            * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
            )
    H_init = 0.5 * ( np.abs(np.random.randn(K,n_frame)) + np.ones((K,n_frame)) )

    # squared mag partial rirs (n_bin, n_channel, n_src)
    Q_init = np.moveaxis(np.abs(partial_rirs)**2, [2], [0])

    W_MU, H_MU, Q_MU, cost = \
        multinmf_conv_mu(np.abs(X)**2, W_init, H_init, Q_init, source_NMF_ind, n_iter=n_iter, fix_Q=True, verbose=True)

    # Computation of the spatial source images
    Im = multinmf_recons_im(X, W_MU, H_MU, Q_MU, source_NMF_ind)

    # Inverse STFT
    for j in range(n_src):
        # channel-wise istft with synthesis window
        ie_MU = []
        for ch in range(n_channel):
            ie_MU.append(
                    pra.istft(Im[:,:,j,ch].T, stft_win_len, stft_win_len // 2, win=window, transform=np.fft.irfft)
                    )

        # write the separated source to a wav file
        out_filename = 'data/Speech/' + 'speech_source_' + str(j) + '_MU.wav'
        wavfile.write(out_filename, fs, np.array(ie_MU).T)

def partial_rir(room, n, freqveq):
    ''' 
    compute the frequency-domain rir based on the n closest image sources
    
    Parameters
    ----------
    room: pyroomacoustics.Room
        The room with sources and microphones
    n: int
        The number of image sources to use
    freqveq: nd-array
        The vector containing all the frequency points to compute

    Returns
    -------
    An nd-array of size M x K x len(freqvec) containing all the transfer
    functions with first index for the microphone, second for the source,
    third for frequency.
    '''

    M = room.mic_array.R.shape[1]
    K = len(room.sources)
    F = freqvec.shape[0]
    partial_rirs = np.zeros((M,K,F), dtype=np.complex)

    mic_array_center = np.mean(room.mic_array.R, axis=1)

    for k, source in enumerate(room.sources):

        # set source ordering to nearest to center of microphone array
        source.set_ordering('nearest', ref_point=mic_array_center)

        sources = source[:n]

        # there is most likely a broadcast way of doing this
        for m in range(M):
            delays = pra.distance(room.mic_array.R[:,m,np.newaxis], sources.images) / c
            partial_rirs[m,k,:] = np.sum(np.exp(-2j * np.pi * delays * freqvec[:,np.newaxis]) / (c * delays) * sources.damping[np.newaxis,:], axis=1) / (4 * np.pi)

    return partial_rirs

if __name__ == '__main__':

    # parameters
    fs = 16000
    nfft = 2048  # supposedly optimal at 16 kHz (Ozerov and Fevote)
    max_order = 10  # max image sources order in simulation

    # convolutive separation parameters
    partial_length = 20  # number of image sources to use in the 'raking'
    n_latent_var = 4    # number of latent variables in the NMF
    stft_win_len = 2048  # supposedly optimal at 16 kHz

    # the speech samples
    r1, speech1 = wavfile.read('data/Speech/fq_sample1.wav')
    speech1 /= np.std(speech1)
    r2, speech2 = wavfile.read('data/Speech/fq_sample2.wav')
    speech2 /= np.std(speech2)
    if r1 != fs or r2 != fs:
        raise ValueError('The speech samples should have the same sample rate as the simulation')

    # a 5 wall room
    floorplan = np.array([[0, 0], [7, 0], [7, 5], [2,5], [0,3]]).T
    room = pra.Room.from_corners(floorplan, fs=fs, absorption=0.15, max_order=max_order)
    room.extrude(4.)  # add the third dimension

    # add two sources
    room.add_source([2, 3.1, 1.8], signal=speech1)
    room.add_source([5.5, 4, 1.7], signal=speech2)

    # now add a few microphones
    mic_locations = np.array([[3.5, 3.5, 1.5], [3.6, 3.5, 1.55], [3.55, 3.7, 1.7]]).T
    room.add_microphone_array(
            pra.MicrophoneArray(mic_locations, fs)
            )

    # simulate propagation (through image source model)
    room.compute_rir()
    room.simulate()

    # compute partial rir
    freqvec = np.fft.rfftfreq(nfft, 1 / fs)
    partial_rirs = partial_rir(room, partial_length, freqvec)

    wavfile.write('data/Speech/two_sources_mix.wav', fs, room.mic_array.signals.T)

    # run NMF
    multinmf_conv_mu_wrapper(room.mic_array.signals.T, partial_rirs, n_latent_var, n_iter=500)

    # show all these nice plots
    plt.show()
