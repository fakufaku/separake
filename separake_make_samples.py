import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import datetime
from itertools import product, combinations
import shutil
import time
import os
import json

from multinmf_conv_mu import multinmf_conv_mu_wrapper
from multinmf_conv_em import multinmf_conv_em_wrapper
from utilities import partial_rir, reverse_simulate, reverse_simulate_all_single_sources
from sim_tools import json_append

from mir_eval.separation import bss_eval_images

base_dir = os.path.abspath(os.path.split(__file__)[0])
print('Base dir', base_dir)
output_dir = "/data/results/"

if not os.path.exists(base_dir+output_dir):
    os.mkdir(base_dir+output_dir)

# output filename format. {} is replaced by date/time
data_dir_format = base_dir + output_dir +'{timestamp}_near_wall_{method}'
data_file_format = '/data_{}.json'  # The {} is replaced by node pid
param_file_format = '/parameters.json'  # We store the parameters in a json file
args_file_format = '/arguments.json'  # We store the arguments list in a json file
error_file_format = '/error_{}.json'  # We store some debug info on failed instances

fs = 16000

# room parameters
max_order = 8  # max image sources order in simulation
floorplan = [ [0, 6, 6, 2, 0],   # x-coordinates
              [0, 0, 5, 5, 3] ]  # y-coordinates
height = 4.
absorption = 0.4
# planar circular array with three microphones and 30 cm inter-mic dist
# placed in bottom right corner of the room
mics_locs = [[ 5.61047449,  5.53282877,  5.32069674],    # x-coordinates
             [ 0.38952551,  0.67930326,  0.46717123],    # y-coordinates
             [ 0.70000000,  0.70000000,  0.70000000] ]  # z-coordinates

speech_files = ['data/Speech/fq_sample3.wav', 'data/Speech/fq_sample2.wav',]

dist_src_mic = [2.5, 4] # Put all sources in donut
min_dist_src_src = 1.  # minimum distance between two sources
n_src_locations = len(speech_files)  # number of different source locations to consider
# optimal gamma set empirically
gamma_opt = {'learn': 0.1, 'anechoic': 10., 0: 10., 1: 0.0001, 2:0., 3:0., 4:0, 5:0, 6:0., 7:0.}
def get_gamma(n_echoes):
    if n_echoes == 'learn':
        return 0.1
    elif n_echoes == 'anechoic':
        return 10.
    elif n_echoes == 0:
        return 0.0001
    elif n_echoes > 0:
        return 0.
    else:
        raise ValueError('Negative number of echoes')

# convolutive separation parameters
dictionary_file = 'W_dictionary_sqmag_mu.npz'
stft_win_len = 2048    # supposedly optimal at 16 kHz (Ozerov and Fevote 2010)
use_dict = True
mu_n_latent_var = 4    # number of latent variables (ignored when dictionary is used)
em_n_latent_var = 4    # number of latent variables (ignored when dictionary is used)
base_dir = base_dir



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Separake it!')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', '--learn', action='store_true', help='Learn the TF from the data')
    group.add_argument('-a', '--anechoic', action='store_true', help='Anechoic conditions')
    group.add_argument('-e', '--echoes', type=int, metavar='N', help='Use %(metavar)s echoes to form the TF')
    parser.add_argument('-m', '--method', type=str, default='mu', choices=['mu','em'], help='The algorithm to use')
    parser.add_argument('-p', '--play', action='store_true', help='Play the signals after separation')
    parser.add_argument('-i', '--iter', type=int, default=200, help='Number of iterations of the algorithm')
    parser.add_argument('-s', '--save', metavar='DIR', type=str, help='Save the audio files to %(metavar)s')
    parser.add_argument('--mono', action='store_true', help='Only save the first channel')
    parser.add_argument('--save_rir', type=str, metavar='DIR', help='Plot and save a typical RIR to %(metavar)s')

    args = parser.parse_args()

    # prepare the dictionary
    if use_dict:
        W_dict = np.load(dictionary_file)['W_dictionary']
        mu_n_latent_var = W_dict.shape[1]  # set by dictionary
        print('Using dictionary with', mu_n_latent_var, 'latent variables')
    else:
        W_dict = None

    # the speech samples
    speech_data = []
    n_speech = len(speech_files)
    for sp_fn in speech_files:
        r, audio = wavfile.read(sp_fn)
        audio /= np.std(audio)
        if r != fs:
            raise ValueError('The speech samples should have the same sample rate as the simulation')
        speech_data.append(audio)

    # a 5 wall room
    fp = np.array(floorplan)
    room = pra.Room.from_corners(fp,
                                 fs=fs,
                                 absorption=absorption,
                                 max_order=max_order)
    # add the third dimension
    room.extrude(height, absorption=absorption)

    # add a few microphones
    mics_locs = np.array(mics_locs)
    n_mics = mics_locs.shape[1]
    for m in range(n_mics):
        room.add_source(mics_locs[:,m])

    # generates sources in the room at random locations
    # but ensure they are too close to microphones
    bbox = np.array(
               [ [min(fp[0]), min(fp[1]), 0],
                 [max(fp[0]), max(fp[1]), height] ] ).T
    n_src_locs = n_src_locations  # number of sources
    sources_locs = np.zeros((3,0))
    while sources_locs.shape[1] < n_src_locs:
        # new candidate location in the bounding box
        new_source = np.random.rand(3, 1) * (bbox[:,1] - bbox[:,0])[:,None] + bbox[:,0,None]
        # check the source are in the room
        is_in_room = room.is_inside(new_source[:,0])

        # check the source is not too close to the microphone
        mic_dist = pra.distance(mics_locs, new_source).min()
        distance_mic_ok = (dist_src_mic[0] < mic_dist and
                            mic_dist < dist_src_mic[1])

        select = is_in_room and distance_mic_ok

        if sources_locs.shape[1] > 0:
            distance_src_ok = (min_dist_src_src
                                < pra.distance(sources_locs, new_source).min())
            select = select and distance_src_ok

        if select:
            sources_locs = np.concatenate([sources_locs, new_source], axis=1)

    source_array = pra.MicrophoneArray(sources_locs, fs)
    room.add_microphone_array(source_array)

    if args.anechoic:
        # 1) We let the room be anechoic and simulate all
        #    microphone signals
        room.max_order = 0  # never reflect!
        room.image_source_model()
        room.compute_rir()
        single_sources = reverse_simulate_all_single_sources(room, speech_data)

        partial_rirs = np.ones((n_mics, n_speech, stft_win_len // 2 + 1))
        gamma = get_gamma('anechoic')

    else:
        # 2) Let the room have echoes and recompute all microphone signals
        room.max_order = max_order
        room.image_source_model()
        room.compute_rir()

        # simulate propagation of sources individually
        # mixing will be done in the simulation loop by simple addition
        # shape of single_sources: (n_speech, n_src_locs, n_samples, n_mics_locs)
        single_sources = reverse_simulate_all_single_sources(room, speech_data)

        # compute partial rir
        # (remove negative partial lengths corresponding to anechoic conditions)
        if args.learn:
            partial_rirs = None
            gamma = get_gamma('learn')
        else:
            freqvec = np.fft.rfftfreq(stft_win_len, 1 / room.fs)
            partial_rirs = np.swapaxes(
                    partial_rir(room, args.echoes + 1, freqvec), 0, 1)
            gamma = get_gamma(args.echoes)

    # mix the signal
    mic_signals = np.zeros(single_sources.shape[-2:])  # (n_samples, n_mics)
    for speech_index in range(n_speech):
            mic_signals += single_sources[speech_index,speech_index,:,:]

    # run the method
    if args.method == 'mu':
        # separate using MU
        sep_sources = multinmf_conv_mu_wrapper(
                mic_signals, n_speech, mu_n_latent_var, stft_win_len,
                partial_rirs=partial_rirs,
                W_dict=W_dict, l1_reg=gamma,
                n_iter=args.iter, verbose=True)
    elif args.method == 'em':
        # separate using EM
        sep_sources = multinmf_conv_em_wrapper(
                mic_signals, partial_rirs,
                em_n_latent_var, n_iter=args.iter,
                A_init=partial_rirs_sources, W_init=W_dict,
                update_a=False, update_w=False,
                verbose=False)
    else:
        raise ValueError('Unknown algorithm {} requested'.format(method))

    n_samples = np.minimum(single_sources.shape[2], sep_sources.shape[1])
    reference_signals = []
    for speech_ind in range(n_speech):
        reference_signals.append(single_sources[speech_ind,speech_ind,:n_samples,:])
    reference_signals = np.array(reference_signals)

    ret = \
            bss_eval_images(reference_signals, sep_sources[:,:n_samples,:])

    print('SDR={} ISR={} SIR={} SAR={}'.format(*ret[:4]))

    mic_norm = 0.7 / np.max(np.abs(mic_signals))
    sep_src_norm = 0.7 / np.max(np.abs(sep_sources))

    if args.play:
        import sounddevice as sd
        sd.play(mic_signals[:,:2] / mic_norm, samplerate=fs, blocking=True)
        for s in range(n_speech):
            sd.play(sep_sources[s,:,:2] / sep_src_norm, samplerate=fs, blocking=True)

    if args.save is not None:

        if args.mono:
            save_mix = mic_signals[:,0]
            save_sep = sep_sources[:,:,0]
        else:
            save_mix = mic_signals
            save_sep = sep_sources

        bnames = [os.path.splitext(os.path.basename(name))[0] for name in speech_files]
        filename = args.save + '/separake_{}_mix_'.format(args.method) + '_'.join(bnames) + '.wav'
        wavfile.write(filename, fs, save_mix)
        for i, name in enumerate(bnames):
            filename = args.save + '/separake_{}_sep_'.format(args.method) + name + '.wav'
            wavfile.write(filename, fs, save_sep[i])

    if args.save_rir is not None:

        n_taps = len(room.rir[0][0])
        import seaborn as sns
        sns.set(style='white', context='paper', font_scale=0.8,
                rc={
                    'figure.figsize': (1.5748, 1.29921),  # 40 x 33 mm
                    'lines.linewidth': 0.5,
                    'font.family': u'Roboto',
                    'font.sans-serif': [u'Roboto Thin'],
                    'text.usetex': False,
                    })
        plt.figure()
        plt.plot(np.arange(n_taps) / room.fs, room.rir[0][0])
        plt.xlabel('Time [s]')
        plt.yticks([])
        sns.despine(left=True, bottom=True)
        plt.tight_layout(pad=0.5)
        plt.savefig(args.save_rir + '/typical_rir.pdf')



