import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from multinmf_conv_mu import multinmf_conv_mu_wrapper
from utilities import partial_rir, reverse_simulate
from sim_tools import json_append

from mir_eval.separation import bss_eval_images

base_dir = os.path.abspath(os.path.split(__file__)[0])
print('Base dir', base_dir)

if not os.path.exists('data'):
    os.mkdir('data')

# output filename format. {} is replaced by date/time
data_dir_format = base_dir + '/data/{}_mu_near_wall'  # The {} is replaced with date
data_file_format = 'data_{}.json'  # The {} is replaced by node pid
param_file_format = '/parameters.json'  # We store the parameters in a json file
args_file_format = '/args.json'  # We store the arguments list in a json file
error_file_format = '/error_{}.json'  # We store some debug info on failed instances

# parameters
parameters = dict(
    fs = 16000,

    # room parameters
    max_order = 8,  # max image sources order in simulation
    floorplan = [[0, 0], [6, 0], [6, 5], [2,5], [0,3]],
    height = 4.,
    absorption = 0.4,
    mics_locs = [ [ 5.50, 5.50, 5.50 ],
                   [ 4.53, 4.55, 4.57 ],
                   [ 0.70, 0.70, 0.70 ] ],

    n_epochs = 3  # number of trials for each parameters combination

    # convolutive separation parameters
    dictionary_file = 'W_dictionary_sqmag.npz',
    mu_n_iter = 200,        # number of iterations of MU algorithm
    stft_win_len = 2048,    # supposedly optimal at 16 kHz (Ozerov and Fevote 2010)
    use_dict = True,
    mu_n_latent_var = 4,    # number of latent variables (ignored when dictionary is used)
    )

# parameters to sweep
partial_lengths = list(range(0,11,2))  # number of image sources to use in the 'raking'
l1_reg = [1e-2,1e-3,1e-4] # only used with a dictionary, automatically set to zero otherwise
seeds = np.random.randint(2**32, size=n_epochs)
args = itertools.product(partial_lengths, l1_reg, seeds)

def parallel_loop(partial_length, gamma, result_file=None,
        stft_win_len=None, fs=None, room=None, src_signals=None, mic_signals=None,
        mu_n_latent_var=None, W_dict=None, mu_n_iter=None, seed=None, **kwargs):
    ''' This the function that should be dumb parallel '''

    # compute partial rir
    freqvec = np.fft.rfftfreq(stft_win_len, 1 / fs)
    partial_rirs = partial_rir(room, partial_length + 1, freqvec)

    partial_rirs_sources = np.swapaxes(
            np.array( [pr for i,pr in enumerate(partial_rirs) if src_signals[i] is not None]),
            0, 1)

    # separate using MU
    sep_sources = multinmf_conv_mu_wrapper(
            mic_signals.T, partial_rirs_sources, 
            mu_n_latent_var, W_dict=W_dict, l1_reg=gamma, 
            n_iter=mu_n_iter, verbose=False, random_seed=seed)

    # compute the metrics
    n_samples = np.minimum(single_sources.shape[1], sep_sources.shape[1])
    ret = \
            bss_eval_images(single_sources[:,:n_samples,:], sep_sources[:,:n_samples,:])

    entry = dict(
            partial_length=partial_length,
            gamma=gamma,
            seed=seed,
            sdr=ret[0].tolist(),
            isr=ret[1].tolist(),
            sir=ret[2].tolist(),
            sar=ret[3].tolist(),
            )

    json_append(result_file, entry)

    return entry


if __name__ == '__main__':

    '''
    In this script we are interested in finding image microphones. Since pyroomacoustics
    has been designed to work with image sources, a simple hack is to reverse the roles
    of sources and microphones.
    '''

    # the speech samples
    r1, speech1 = wavfile.read('data/Speech/fq_sample1.wav')
    speech1 /= np.std(speech1)
    r2, speech2 = wavfile.read('data/Speech/fq_sample2.wav')
    speech2 /= np.std(speech2)
    if r1 != fs or r2 != fs:
        raise ValueError('The speech samples should have the same sample rate as the simulation')

    # a 5 wall room
    room = pra.Room.from_corners(np.array(parameters['floorplan']).T,
                                 fs=parameters['fs'], 
                                 absorption=parameters['absorption'],
                                 max_order=parameters['max_order'])
    # add the third dimension
    room.extrude(parameters['height'], absorption=parameters['absorption'])

    # add two sources
    K = 10  # number of sources
    sources_locs = np.concatenate((
            pra.linear_2D_array([2.05, 2.5], K, np.pi / 2, 0.5),
            1.7 * np.ones((1, K))
            ))
    source_array = pra.MicrophoneArray(sources_locs, fs)
    room.add_microphone_array(source_array)

    # now add a few microphones
    mics_locs = np.array(parameters['mics_locs'])
    for m in range(mics_locs.shape[1]):
        room.add_source(mics_locs[:,m])

    # compute the RIR between sources and microphones
    room.compute_rir()

    # simulate propagation with two sources
    src_signals = [None] * K
    src_signals[0] = speech1
    src_signals[K-1] = speech2
    mic_signals = reverse_simulate(room, src_signals)

    # simulate also the sources separately for comparison
    single_sources = []
    for i,s in enumerate(src_signals):
        if s is None:
            continue
        feed = [None] * K
        feed[i] = s
        single_sources.append(reverse_simulate(room, feed, length=mic_signals.shape[1]))
    single_sources = np.swapaxes(np.array(single_sources), 1, 2)

    # prepare the dictionary
    if use_dict:
        W_dict = np.load('W_dictionary_sqmag.npz')['W_dictionary']
        mu_n_latent_var = W_dict.shape[1]  # set by dictionary
        print('Using dictionary with', mu_n_latent_var, 'latent variables')
    else:
        W_dict = None
        l1_reg = 0.

    # run simulation
    scores = dict(
        sdr = np.zeros((len(partial_lengths), len(l1_reg), n_epochs)),
        isr = np.zeros((len(partial_lengths), len(l1_reg), n_epochs)),
        sir = np.zeros((len(partial_lengths), len(l1_reg), n_epochs)),
        sar = np.zeros((len(partial_lengths), len(l1_reg), n_epochs)),
        )

    seeds = np.random.randint(2**32, size=n_epochs)

    for epoch, seed in zip(range(n_epochs), seeds):
        print('Epoch', epoch)
        for index, partial_length in enumerate(partial_lengths):

            for g, gamma in enumerate(l1_reg):
                print('  Number of image microphones:', partial_length)


    # plot the results
    plt.figure()
    for i, metric in enumerate(['sdr', 'isr', 'sir', 'sar']):
        plt.subplot(2,2,i+1)
        for g in range(len(l1_reg)):
            plt.plot(partial_lengths, scores[metric][:,g,:].mean(axis=-1))
        plt.legend(l1_reg)
        plt.xlabel('number of image microphones')
        plt.ylabel(metric)

    plt.tight_layout()
    plt.show()

