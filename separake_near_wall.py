import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra

from utilities import partial_rir, reverse_simulate

if __name__ == '__main__':

    '''
    In this script we are interested in finding image microphones. Since pyroomacoustics
    has been designed to work with image sources, a simple hack is to reverse the roles
    of sources and microphones.
    '''

    # parameters
    fs = 16000
    nfft = 2048  # supposedly optimal at 16 kHz (Ozerov and Fevote)
    max_order = 10  # max image sources order in simulation

    # convolutive separation parameters
    partial_length = 2  # number of image sources to use in the 'raking'
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
    K = 10  # number of sources
    sources_locs = np.concatenate((
            pra.linear_2D_array([2.05, 2.5], K, np.pi / 2, 0.5),
            1.7 * np.ones((1, K))
            ))
    source_array = pra.MicrophoneArray(sources_locs, fs)
    room.add_microphone_array(source_array)

    # now add a few microphones
    mics_locs = np.array([ [ 6.50, 6.50, 6.50 ],
                           [ 2.53, 2.55, 2.57 ],
                           [ 1.70, 1.70, 1.70 ] ])
    for m in range(mics_locs.shape[1]):
        room.add_source(mics_locs[:,m])

    # compute the RIR between sources and microphones
    room.compute_rir()


    # simulate propagation with two sources
    src_signals = [None] * K
    src_signals[1] = speech1
    src_signals[4] = speech2
    signals = reverse_simulate(room, src_signals)

    # compute partial rir
    freqvec = np.fft.rfftfreq(nfft, 1 / fs)
    partial_rirs = partial_rir(room, partial_length, freqvec)



