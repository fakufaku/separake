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

if not os.path.exists('data'):
    os.mkdir('data')

# output filename format. {} is replaced by date/time
data_dir_format = base_dir + '/data/{timestamp}_near_wall_{method}'
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

    master_seed = 0xDEADBEEF,  # seed of the random number generator

    min_dist_src_mic = 2., # Impose a minimum distance of between sources and microphones [in meters]
    n_src_locations = 10,  # number of different source locations to consider
    n_epochs = 1,          # number of trials for each parameters combination

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

# initialize the random number generator
# this makes the simulation repeatable
np.random.seed(parameters['master_seed'])

# parameters to sweep
#####################

# the active source indices
n_src = len(parameters['speech_files'])
src_locs_ind = list(combinations(range(parameters['n_src_locations']), n_src))  

# number of image sources to use in the 'raking', or -1 for anechoic conditions
partial_lengths = [-1,0,1,2,4,7]  

# only used with a dictionary, automatically set to zero otherwise
l1_reg = [100, 1., 1e-2, 1e-4] 

# seed to enforce same random intialization for all run of the algorithm
# under different parameters
seeds = np.random.randint(2**32, size=parameters['n_epochs']).tolist()

# cartesian products of all the arguments
arguments = list(product(src_locs_ind, partial_lengths, l1_reg, seeds))

# This is used for debugging.
# we want to use mkl acceleration when running in
# serial mode, but not on the cluster
use_mkl = False

def parallel_loop(args):
    ''' This is the function that should be dumb parallel '''

    # expand positional arguments
    src_locs_ind, partial_length, gamma, seed = args

    # now the keyword arguments
    result_file = parameters['result_file']
    stft_win_len = parameters['stft_win_len']
    fs = parameters['fs']
    room = parameters['room']
    partial_rirs = parameters['partial_rirs']
    single_sources = parameters['single_sources']
    single_sources_anechoic = parameters['single_sources_anechoic']
    mu_n_latent_var = parameters['mu_n_latent_var']
    em_n_latent_var = parameters['em_n_latent_var']
    W_dict = parameters['W_dict']
    em_n_iter = parameters['em_n_iter']
    mu_n_iter = parameters['mu_n_iter']
    base_dir = parameters['base_dir']
    method = parameters['method']

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

    # select between echoic and anechoic signals
    if partial_length >= 0:
        clean_sources = single_sources
    else:
        # anechoic propagation
        clean_sources = single_sources_anechoic

    # mix the sources
    mic_signals = np.zeros(clean_sources.shape[-2:])  # (n_samples, n_mics)
    for speech_index, loc_index in enumerate(src_locs_ind):
            mic_signals += clean_sources[speech_index,loc_index,:,:]

    # shape (n_mics, n_src, n_bins)
    if partial_length >= 0:
        partial_rirs_sources = np.swapaxes(
                partial_rirs[partial_length][src_locs_ind,:,:], 0, 1)
    else:
        # in anechoic conditions, we have flat responses everywhere
        partial_rirs_sources = np.swapaxes(
                partial_rirs[0][src_locs_ind,:,:], 0, 1)

    if method == 'mu':
        # separate using MU
        sep_sources = multinmf_conv_mu_wrapper(
                mic_signals, partial_rirs_sources,
                mu_n_latent_var, W_dict=W_dict, l1_reg=gamma,
                n_iter=mu_n_iter, verbose=False, random_seed=seed)
    elif method == 'em':
        # separate using EM
        sep_sources = multinmf_conv_em_wrapper(
                mic_signals, partial_rirs_sources,
                em_n_latent_var, W_init=W_dict,
                n_iter=em_n_iter, verbose=False)
    else:
        raise ValueError('Unknown algorithm {} requested'.format(method))

    # compute the metrics
    n_samples = np.minimum(clean_sources.shape[2], sep_sources.shape[1])

    reference_signals = []
    for speech_ind, loc_ind in enumerate(src_locs_ind):
        reference_signals.append(clean_sources[speech_ind,loc_ind,:n_samples,:]) 
    reference_signals = np.array(reference_signals)

    ret = \
            bss_eval_images(reference_signals, sep_sources[:,:n_samples,:])

    entry = dict(
            src_locs_ind=src_locs_ind,
            partial_length=partial_length,
            algorithm=method,
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
        data_dir_name = data_dir_format.format(timestamp=date,method=parameters['method'])

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
    n_speech = len(parameters['speech_files'])
    for sp_fn in parameters['speech_files']:
        r, audio = wavfile.read(sp_fn)
        audio /= np.std(audio)
        if r != parameters['fs']:
            raise ValueError('The speech samples should have the same sample rate as the simulation')
        speech_data.append(audio)

    # a 5 wall room
    room = pra.Room.from_corners(np.array(parameters['floorplan']),
                                 fs=parameters['fs'], 
                                 absorption=parameters['absorption'],
                                 max_order=parameters['max_order'])
    # add the third dimension
    room.extrude(parameters['height'], absorption=parameters['absorption'])

    # add a few microphones
    mics_locs = np.array(parameters['mics_locs'])
    n_mics = mics_locs.shape[1]
    for m in range(n_mics):
        room.add_source(mics_locs[:,m])

    # generates sources in the room at random locations
    # but ensure they are too close to microphones
    fp = parameters['floorplan']
    bbox = np.array(
               [ [min(fp[0]), min(fp[1]), 0],
                 [max(fp[0]), max(fp[1]), parameters['height']] ] ).T
    n_src_locs = parameters['n_src_locations']  # number of sources
    sources_locs = np.zeros((3,0))
    while sources_locs.shape[1] < n_src_locs:
        # new candidate locations in the bounding box
        new_sources = np.random.rand(3, n_src_locs - sources_locs.shape[1]) * (bbox[:,1] - bbox[:,0])[:,None] + bbox[:,0,None]

        # check the sources are in the room
        is_in_room = [room.is_inside(src) for src in new_sources.T]

        # check the sources are not too close to the microphone
        distance_ok = parameters['min_dist_src_mic'] < pra.distance(mics_locs, new_sources).min(axis=0)

        select = np.logical_and(is_in_room, distance_ok)
        sources_locs = np.concatenate([sources_locs, new_sources[:,select]], axis=1)

    source_array = pra.MicrophoneArray(sources_locs, parameters['fs'])
    room.add_microphone_array(source_array)

    # 1) We let the room be anechoic and simulate all
    #    microphone signals
    room.max_order = 0  # never reflect!
    room.image_source_model()
    room.compute_rir()
    single_sources_anechoic = reverse_simulate_all_single_sources(room, speech_data)


    # 2) Let the room have echoes and recompute all microphone signals
    room.max_order = parameters['max_order']
    room.image_source_model()
    room.compute_rir()

    # simulate propagation of sources individually
    # mixing will be done in the simulation loop by simple addition
    # shape of single_sources: (n_speech, n_src_locs, n_samples, n_mics_locs)
    single_sources = reverse_simulate_all_single_sources(room, speech_data)

    # compute partial rir
    # (remove negative partial lengths corresponding to anechoic conditions)
    freqvec = np.fft.rfftfreq(parameters['stft_win_len'], 1 / room.fs)
    partial_rirs = dict(
        [(L, partial_rir(room, L + 1, freqvec)) for L in partial_lengths if L >= 0])

    parameters['partial_rirs'] = partial_rirs
    parameters['source_locations'] = sources_locs
    parameters['single_sources'] = single_sources
    parameters['single_sources_anechoic'] = single_sources_anechoic
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
            out.append(parallel_loop(ag))

    else:
        import ipyparallel as ip

        print('Using ipyparallel processing.')

        # Start the parallel processing
        c = ip.Client(profile=ipcluster_profile)
        NC = len(c.ids)
        print(NC, 'workers on the job')
        # Push the global config to the workers
        c[:].push(dict(parameters=parameters, use_mkl=False), block=True)

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

        print('Show all output from nodes, if any:')
        ar.display_outputs()

        all_loops = time.time() - then
        print('Total actual processing time:', all_loops)

    print('Saved data to folder: ' + data_dir_name)
