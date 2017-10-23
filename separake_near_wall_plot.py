from __future__ import division, print_function

import sys, argparse, copy, os, re
import numpy as np
import pandas as pd
import json

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
            description='Plot the data simulated by separake_near_wall')
    parser.add_argument('-p', '--pickle', action='store_true', 
            help='Read the aggregated data table from a pickle cache')
    parser.add_argument('-s', '--show', action='store_true',
            help='Display the plots at the end of data analysis')
    parser.add_argument('dir', type=str,
            help='The directory containing the simulation output files.')

    cli_args = parser.parse_args()
    data_dir = cli_args.dir
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    # Read in the parameters
    with open(data_dir + '/parameters.json', 'r') as f:
        parameters = json.load(f)

    with open(data_dir + '/arguments.json', 'r') as f:
        args = json.load(f)

    # algorithms to take in the plot
    metrics = ['SDR', 'SIR', 'ISR', 'SAR']

    # check if a pickle file exists for these files
    pickle_file = data_dir + '/dataframe.pickle'

    if os.path.isfile(pickle_file) and pickle_flag:
        print('Reading existing pickle file...')
        # read the pickle file
        perf = pd.read_pickle(pickle_file)

    else:

        # reading all data files in the directory
        records = []
        for file in os.listdir(data_dir):
            if file.startswith('data_') and file.endswith('.json'):
                with open(os.path.join(data_dir, file), 'r') as f:
                    records += json.load(f)

        # build the data table line by line
        print('Building table')
        columns = ['n_echos','gamma','seed'] + metrics
        table = []

        for record in records:
            table.append(
                    [ record['partial_length'], record['gamma'], record['seed'] ]
                    + [np.mean(record[m.lower()]) for m in metrics]
                    )
           
        # create a pandas frame
        print('Making PANDAS frame...')
        df = pd.DataFrame(table, columns=columns)

        # turns out all we need is the follow pivoted table
        perf = pd.pivot_table(df, values=metrics, index='n_echos', columns='gamma', aggfunc=np.mean)

        perf.to_pickle(pickle_file)

    # Draw the figure
    print('Plotting...')

    sns.set(style='whitegrid')
    sns.plotting_context(context='paper', font_scale=1.)

    sns.set(style='whitegrid', context='paper', font_scale=1.2,
            rc={
                'figure.figsize':(3.5,3.15), 
                'lines.linewidth':1.,
                'font.family': 'sans-serif',
                'font.sans-serif': [u'Helvetica'],
                'text.usetex': False,
                })

    # plot the results
    plt.figure()
    for i, metric in enumerate(metrics):
        ax = plt.subplot(2,2,i+1)
        perf[metric].plot(ax=ax, legend=False)
        if i == 3:
            leg = plt.legend(perf[metric].columns, 
                            title='$\gamma$',
                            frameon=True, labelspacing=0.5, 
                            framealpha=0.8, loc=6,
                            bbox_to_anchor=[1.05, 1.2])
            leg.get_frame().set_linewidth(0.0)
        plt.ylabel(metric)
        plt.xlabel('Number of echoes')

    sns.despine(offset=10, trim=False, left=True, bottom=True)

    plt.tight_layout(pad=0.5)

    plt.savefig('figures/separake_near_wall_mu.pdf',
            bbox_extra_artists=(leg,), bbox_inches='tight')

    if plot_flag:
        plt.show()

