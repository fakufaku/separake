import numpy as np
import pyroomacoustics as pra
from itertools import product

# get the speed of sound from pyroomacoustics
c = pra.constants.get('c')

def partial_rir(room, n, freqvec):
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
    functions with first index for the microphones, second for the sources,
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

def reverse_simulate_all_single_sources(room, speech_data):
    '''
    Simulate playing each signal in speech_data at each possible source locations

    Parameters
    ----------
    room: pyroomacoustics.Room
        the room
    speech_data: list of array_like
        list containing all the speech samples to simulate

    Returns
    -------
    An ndarray of shape (n_speech_data, n_src, n_samples, n_channel)
    '''

    n_src_locs = room.mic_array.R.shape[1]  # mics are sources
    n_mics = len(room.sources)              # sources are mics

    max_length_speech = np.max([s.shape[0] for s in speech_data])
    max_length_rir = np.max([len(room.rir[i][j]) for i, j in product(range(n_src_locs), range(n_mics))])
    n_samples = max_length_speech + max_length_rir - 1
    single_sources = []
    for sample in speech_data:
        single_sources.append([])
        for i in range(n_src_locs):
            feed = [None] * n_src_locs
            feed[i] = sample
            single_sources[-1].append(reverse_simulate(room, feed, length=n_samples))
    # final shape of single_sources after axes swap: (n_speech, n_src_locs, n_samples, n_mics_locs)
    # the swap of axis is to make the shape compatible with bss_eval
    return np.swapaxes(single_sources, 2, 3)

def reverse_simulate(room, source_signals, delays=None, length=None):
    '''
    Simulate playing a source at microphones locations and recording at the source locations.

    Parameters
    ----------
    room: pyroomacoustics.Room
        The room
    source_signals: list
        A list of signals to play for each microphone in the room
    delays: array_like
        Possible delays associated with each signal to play
    length: int
        The length in samples of the output signal. If the actual signal is
        longer, it is truncated. If it is shorter, it is zero-padded.

    Returns
    -------
    An ndarray that contains in its rows the recorded signals at the source locations.
    '''

    # import convolution routine
    from scipy.signal import fftconvolve

    # Throw an error if we are missing some hardware in the room
    if (len(room.sources) is 0):
        raise ValueError('There are no sound sources in the room.')
    if (room.mic_array is None):
        raise ValueError('There is no microphone in the room.')

    # compute RIR if necessary
    if room.rir is None or len(room.rir) == 0:
        room.compute_rir()

    if delays is None:
        delays = np.zeros(len(source_signals))

    delays_samples = np.floor(np.array(delays) * room.fs).astype(np.int)

    if len(delays_samples) != room.mic_array.M or len(source_signals) != room.mic_array.M:
        raise ValueError('The same number of signals than microphones is required')

    # number of mics and sources
    M = room.mic_array.M   # number locations where to play the signals
    S = len(room.sources)  # number locations where we record

    # compute the maximum signal length
    from itertools import product
    max_len_rir = np.array([len(room.rir[i][j])
                            for i, j in product(range(M), range(S))]).max()
    max_sig_len = np.array([len(source_signals[i]) + delays_samples[i] for i in range(M) if source_signals[i] is not None]).max()
    L = int(max_len_rir) + int(max_sig_len) - 1
    if L % 2 == 1:
        L += 1

    # enforce length parameter if given
    if length is not None:
        L = length

    # the array that will receive all the signals
    signals = np.zeros((S, L))

    # compute the signal at every microphone in the array
    for m in np.arange(M):
        sig = source_signals[m]
        if sig is None:
            continue
        for s in np.arange(S):
            rx = signals[s]
            d = delays_samples[m]
            h = room.rir[m][s]

            if d > L:
                continue
            elif d + len(sig) + len(h) - 1 > L:
                d2 = L
            else:
                d2 = d + len(sig) + len(h) - 1
            rx[d:d2] += fftconvolve(h, sig[:d2 - d])

        # add white gaussian noise if necessary
        if room.sigma2_awgn is not None:
            rx += np.random.normal(0., np.sqrt(room.sigma2_awgn), rx.shape)

    return signals

