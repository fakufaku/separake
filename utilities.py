import numpy as np
import pyroomacoustics as pra

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


def reverse_simulate(room, source_signals, delays=None):
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
    max_sig_len = np.array([len(source_signals[i]) + delays_samples[i] for i in range(S) if source_signals[i] is not None]).max()
    L = int(max_len_rir) + int(max_sig_len) - 1
    if L % 2 == 1:
        L += 1

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
            rx[d:d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

        # add white gaussian noise if necessary
        if room.sigma2_awgn is not None:
            rx += np.random.normal(0., np.sqrt(room.sigma2_awgn), rx.shape)

    return signals

