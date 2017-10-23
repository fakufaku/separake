import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from functools import partial
import datetime
import itertools
import shutil
import time
import os
import json

from multinmf_conv_mu import multinmf_conv_mu_wrapper
from utilities import partial_rir, reverse_simulate
from sim_tools import json_append

from mir_eval.separation import bss_eval_images

base_dir = os.path.abspath(os.path.split(__file__)[0])
print('Base dir', base_dir)

if not os.path.exists('data'):
    os.mkdir('data')

# output filename format. {} is replaced by date/time
data_dir_format = base_dir + '/data/{}_mu_near_wall'
data_file_format = '/data_{}.json'  # The {} is replaced by node pid
param_file_format = '/parameters.json'  # We store the parameters in a json file
args_file_format = '/arguments.json'  # We store the arguments list in a json file
error_file_format = '/error_{}.json'  # We store some debug info on failed instances

# parameters
parameters = dict(
    fs = 16000,

    # room parameters
    max_order = 8,  # max image sources order in simulation
    floorplan = [ [0, 6, 6, 2, 0],   # x-coordinates
                  [0, 0, 5, 5, 3] ],  # y-coordinates
    height = 4.,
    absorption = 0.4,
    mics_locs = [ [ 5.50, 5.50, 5.50 ],    # x-coordinates
                  [ 4.53, 4.55, 4.67 ],    # y-coordinates
                  [ 0.70, 0.70, 0.70 ] ],  # z-coordinates

    speech_files = ['data/Speech/fq_sample3.wav', 'data/Speech/fq_sample2.wav',],

    n_epochs = 1,         # number of trials for each parameters combination
    n_src_locations = 10,  # number of different source locations to consider

    # convolutive separation parameters
    method = "mu",          # solving method: mu or em
    dictionary_file = 'W_dictionary_sqmag_mu.npz', #or 'W_dictionary_em
    em_n_iter = 100,        # number of iterations of MU algorithm
    mu_n_iter = 200,        # number of iterations of MU algorithm
    stft_win_len = 2048,    # supposedly optimal at 16 kHz (Ozerov and Fevote 2010)
    use_dict = True,
    mu_n_latent_var = 4,    # number of latent variables (ignored when dictionary is used)
    em_n_latent_var = 4,    # number of latent variables (ignored when dictionary is used)
    base_dir = base_dir,
    )

# parameters to sweep

partial_lengths = list(range(0,11,1))  # number of image sources to use in the 'raking'
l1_reg = [1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4] # only used with a dictionary, automatically set to zero otherwise
seeds = np.random.randint(2**32, size=parameters['n_epochs']).tolist()
arguments = list(itertools.product(partial_lengths, l1_reg, seeds))

# This is used for debugging.
# we want to use mkl acceleration when running in
# serial mode, but not on the cluster
use_mkl = False

def parallel_loop(args):
    ''' This is the function that should be dumb parallel '''
    # expand positional arguments
    partial_length, gamma, seed = args

    # now the keyword arguments
    result_file=parameters['result_file']
    stft_win_len=parameters['stft_win_len']
    fs=parameters['fs']
    room=parameters['room']
    src_signals=parameters['src_signals']
    mic_signals=parameters['mic_signals']
    single_sources=parameters['single_sources']
    mu_n_latent_var=parameters['mu_n_latent_var']
    em_n_latent_var=parameters['em_n_latent_var']
    W_dict=parameters['W_dict']
    em_n_iter==parameters['em_n_iter']
    mu_n_iter=parameters['mu_n_iter']
    base_dir=parameters['base_dir']
    method=parameters["method"]

    # make sure base dir is in path
    import sys, os
    if base_dir not in sys.path:
        sys.path.append(base_dir)

    import numpy as np
    from mir_eval.separation import bss_eval_images
    from multinmf_conv_mu import multinmf_conv_mu_wrapper
    from utilities import partial_rir
    from sim_tools import json_append

    try:
        import mkl as mkl_service
        # for such parallel processing, it is better
        # to deactivate multithreading in mkl
        if not use_mkl:
            mkl_service.set_num_threads(1)
    except ImportError:
        pass

    # compute partial rir
    freqvec = np.fft.rfftfreq(stft_win_len, 1 / fs)
    partial_rirs = partial_rir(room, partial_length + 1, freqvec)

    partial_rirs_sources = np.swapaxes(
            np.array( [pr for i,pr in enumerate(partial_rirs) if src_signals[i] is not None]),
            0, 1)

    sep_sources = dict()

    # separate using MU
    if method is "mu":
        sep_sources = multinmf_conv_mu_wrapper(
                mic_signals.T, partial_rirs_sources,
                mu_n_latent_var, W_dict=W_dict, l1_reg=gamma,
                n_iter=mu_n_iter, verbose=False, random_seed=seed)

    # separate using EM
    if method is "em":
        sep_sources = multinmf_conv_em_wrapper(
                mic_signals.T, partial_rirs_sources,
                em_n_latent_var, W_init=W_dict,
                n_iter=em_n_iter, verbose=False)

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

    filename = result_file.format(os.getpid())
    json_append(filename, entry)

    return entry


if __name__ == '__main__':

    '''
    In this script we are interested in finding image microphones. Since pyroomacoustics
    has been designed to work with image sources, a simple hack is to reverse the roles
    of sources and microphones.
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Separake it!')
    parser.add_argument('-d', '--dir', type=str, help='directory to store sim results')
    parser.add_argument('-p', '--profile', type=str, help='ipython profile of cluster')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--serial', action='store_true')

    args = parser.parse_args()
    ipcluster_profile = args.profile
    test_flag = args.test
    serial_flag = args.serial
    data_dir_name = None

    # Save the result to a directory
    if data_dir_name is None:
        date = time.strftime("%Y%m%d-%H%M%S")
        data_dir_name = data_dir_format.format(date)

    # create directory if it doesn't exist
    try:
        os.mkdir(data_dir_name)
    except:
        if not os.path.exists(data_dir_name):
            raise ValueError('Couldn''t create the data directory')
        else:
            pass

    # this is the file name template to store the results
    parameters['result_file'] = data_dir_name + data_file_format

    # prepare the dictionary
    if parameters['use_dict']:
        W_dict = np.load(parameters['dictionary_file'])['W_dictionary']
        mu_n_latent_var = W_dict.shape[1]  # set by dictionary
        print('Using dictionary with', mu_n_latent_var, 'latent variables')
        parameters['mu_n_latent_var'] = mu_n_latent_var
        # save a copy of the dictionary to the sim directory
        copyfilename = data_dir_name + '/' + os.path.basename(parameters['dictionary_file'])
        shutil.copyfile(parameters['dictionary_file'], copyfilename)
    else:
        W_dict = None

    # Save the parameters in a json file
    parameters_file = data_dir_name + param_file_format
    with open(parameters_file, "w") as f:
        json.dump(parameters, f)
        f.close()

    # Save the arguments in a json file
    args_file = data_dir_name + args_file_format
    with open(args_file, "w") as f:
        json.dump(arguments, f)
        f.close()

    # the speech samples
    speech_data = []
    for sp_fn in parameters['speech_files']:
        r, speech = wavfile.read(sp_fn)
        speech /= np.std(speech)
        if r != parameters['fs']:
            raise ValueError('The speech samples should have the same sample rate as the simulation')
        speech_data.append(speech)

    # a 5 wall room
    room = pra.Room.from_corners(np.array(parameters['floorplan']),
                                 fs=parameters['fs'], 
                                 absorption=parameters['absorption'],
                                 max_order=parameters['max_order'])
    # add the third dimension
    room.extrude(parameters['height'], absorption=parameters['absorption'])

    # generates sources in the room at random locations
    fp = parameters['floorplan']
    bbox = np.array(
               [ [min(fp[0]), min(fp[1]), 0],
                 [max(fp[0]), max(fp[1]), parameters['height']] ] ).T
    K = parameters['n_src_locations']  # number of sources
    sources_locs = np.zeros((3,0))
    while sources_locs.shape[1] < K:
        new_sources = np.random.rand(3, K - sources_locs.shape[1]) * (bbox[:,1] - bbox[:,0])[:,None] + bbox[:,0,None]
        is_in_room = [room.is_inside(src) for src in new_sources.T]
        sources_locs = np.concatenate([sources_locs, new_sources[:,is_in_room]], axis=1)

    source_array = pra.MicrophoneArray(sources_locs, parameters['fs'])
    room.add_microphone_array(source_array)

    # now add a few microphones
    mics_locs = np.array(parameters['mics_locs'])
    for m in range(mics_locs.shape[1]):
        room.add_source(mics_locs[:,m])

    # compute the RIR between sources and microphones
    room.compute_rir()

    # simulate propagation with two sources
    src_signals = [None] * K
    src_signals[0] = speech_data[0]
    src_signals[K-1] = speech_data[1]
    mic_signals = reverse_simulate(room, src_signals)

    # simulate also the sources separately for comparison
    max_length_speech = np.max([s.shape[0] for s in speech_data])
    max_rir_length = np.max([ max([ len(rir) for rir in rirs ]) for rirs in room.rir ])
    max_length = max_length_speech + max_rir_length - 1
    single_sources = []
    for i,s in enumerate(src_signals):
        if s is None:
            continue
        feed = [None] * K
        feed[i] = s
        single_sources.append(reverse_simulate(room, feed, length=mic_signals.shape[1]))
    single_sources = np.swapaxes(np.array(single_sources), 1, 2)

    import pdb
    pdb.set_trace()


    parameters['source_locations'] = sources_locs
    parameters['src_signals'] = src_signals
    parameters['mic_signals'] = mic_signals
    parameters['single_sources'] = single_sources
    parameters['W_dict'] = W_dict
    parameters['room'] = room

    # There is the option to only run one loop for test
    if test_flag:
        print('Running one test loop only.')
        arguments = arguments[:1]

    # Main processing loop
    if serial_flag:
        print('Running everything in a serial loop.')
        use_mkl = True
        # Serial processing
        out = []
        for ag in arguments:
            out.append(parallel_loop(ag, **parameters))

    else:
        import ipyparallel as ip

        print('Using ipyparallel processing.')

        # Start the parallel processing
        c = ip.Client(profile=ipcluster_profile)
        NC = len(c.ids)
        print(NC, 'workers on the job')
        # Push the global config to the workers
        c[:].push(dict(parameters=parameters, use_mkl=False))

        # use a load balanced view
        lbv = c.load_balanced_view()

        # record start timestamp
        then = time.time()
        start_time = datetime.datetime.now()

        # dispatch to workers
        ar = lbv.map_async(parallel_loop, arguments)

        # prepare the status line
        n_tasks = len(arguments)
        digits = int(np.log10(n_tasks) + 1)
        dformat = '{:' + str(digits) + 'd}'
        status_line = dformat + '/' + dformat + ' tasks done. Forecast end {:>20s}'

        forecast = 'NA'
        while not ar.done():

            n_remaining = n_tasks - ar.progress

            if ar.progress > NC and n_remaining > NC:

                ellapsed = time.time() - then

                # estimate remaining time
                rate = ar.progress / ellapsed  # tasks per second
                delta_finish_min = int(rate * n_remaining / 60) + 1

                end_date = start_time + datetime.timedelta(minutes=delta_finish_min)
                forecast = end_date.strftime('%Y-%m-%d %H:%M:%S')

            print(status_line.format(ar.progress, n_tasks, forecast), end='\r')

            time.sleep(1)

        all_loops = time.time() - then
        print('Total actual processing time:', all_loops)

    print('Saved data to folder: ' + data_dir_name)
