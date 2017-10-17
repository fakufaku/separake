
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

from multinmf_conv_mu import multinmf_conv_mu_wrapper
from multinmf_recons_im import multinmf_recons_im

from utilities import partial_rir

# get the speed of sound from pyroomacoustics
c = pra.constants.get('c')

if __name__ == '__main__':

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
    room.add_source([2, 1.5, 1.8], signal=speech1)
    room.add_source([5.5, 4, 1.7], signal=speech2)

    # now add a few microphones
    mic_locations = np.array([[3.65, 3.5, 1.5], [3.6, 3.5, 1.55], [3.55, 3.7, 1.7]]).T
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
    sep_sources = multinmf_conv_mu_wrapper(room.mic_array.signals.T, partial_rirs, n_latent_var, n_iter=500)

    # Plots

    plt.figure()
    plt.subplot(1,2,1)
    plt.specgram(speech1, Fs=r1, NFFT=nfft)
    plt.subplot(1,2,2)
    plt.specgram(speech2, Fs=r2, NFFT=nfft)
    plt.title('Original sources')

    plt.figure()
    for j,s in enumerate(sep_sources):
        # write the separated source to a wav file
        out_filename = 'data/Speech/' + 'speech_source_' + str(j) + '_MU.wav'
        wavfile.write(out_filename, room.fs, s)

        # show spectrogram
        plt.subplot(1,2,j+1)
        plt.specgram(s[:,0], Fs=room.fs, NFFT=nfft)
    plt.title('Reconstructed sources')




    # show all these nice plots
    plt.show()
