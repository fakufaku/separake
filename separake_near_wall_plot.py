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
    parser.add_argument('dirs', type=str, nargs='+', metavar='DIR',
            help='The directory containing the simulation output files.')

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    # metrics to take in the plot
    metrics = ['SDR', 'SIR', 'ISR', 'SAR']
    columns = ['n_echoes','seed','Speaker gender'] + metrics

    parameters = dict()
    args = []
    df = None

    for i, data_dir in enumerate(cli_args.dirs):

        print('Reading in', data_dir)

        # Read in the parameters
        with open(data_dir + '/parameters.json', 'r') as f:
            parameters_part = json.load(f)

            for key, val in parameters_part.items():
                if key in parameters and val != parameters[key]:
                    print('Warning ({}): {}={} (vs {} in {})'.format(data_dir, 
                        key, val, parameters[key], cli_args.dirs[0]))

            else:
                if i > 0:
                    print('Warning ({}): parameter {}={} was not present before'.format(data_dir, key, val))
                parameters[key] = val

        with open(data_dir + '/arguments.json', 'r') as f:
            args.append(json.load(f))

        # check if a pickle file exists for these files
        pickle_file = data_dir + '/dataframe.pickle'

        if os.path.isfile(pickle_file) and pickle_flag:
            print('Reading existing pickle file...')
            # read the pickle file
            df_part = pd.read_pickle(pickle_file)

        else:

            # reading all data files in the directory
            records = []
            for file in os.listdir(data_dir):
                if file.startswith('data_') and file.endswith('.json'):
                    with open(os.path.join(data_dir, file), 'r') as f:
                        records += json.load(f)

            # build the data table line by line
            print('  Building table')
            table = []

            src_ind_2_label = {0: 'Female', 1:'Male'}
            for record in records:
                if record['partial_length'] == 'learn':
                    record['partial_length'] = -2
                if record['partial_length'] == 'anechoic':
                    record['partial_length'] = -1

                for src_index in range(len(record['sdr'])):
                    table.append(
                            [ record['partial_length'], record['seed'], src_ind_2_label[src_index], ]
                            + [record[m.lower()][src_index] for m in metrics]
                            )
               
            # create a pandas frame
            print('  Making PANDAS frame...')
            df_part = pd.DataFrame(table, columns=columns)
            df_part.to_pickle(pickle_file)

        if df is None:
            df = df_part
        else:
            df = pd.concat([df, df_part], ignore_index=True)

    index = np.sort(df.n_echoes.unique()).tolist()
    index_str = [str(i) for i in index]
    if -2 in index:
        index_str[index.index(-2)] = 'learn'
    if -1 in index:
        index_str[index.index(-1)] = 'anechoic'

    # Draw the figure
    print('Plotting...')

    # get a basename for the plot
    plot_basename = ('figures/' + cli_args.dirs[0].rstrip('/').split('/')[-1])

    # compute median and save to pickle
    df_median = df[['n_echoes','SDR','SIR']].groupby('n_echoes').median()
    df_median.to_pickle(plot_basename + '_median.pickle')
    df.to_pickle(plot_basename + '_dataframe.pickle')

    # Now plot the final figure, that should be much nicer and go in the paper
    #newdf = df.replace({'n_echoes': {-2:'learn', -1:'anechoic'}})
    newdf = df.replace({'n_echoes': dict(zip(index, index_str))})
    mdf = newdf.melt(
            id_vars=['n_echoes','Speaker gender'], 
            value_vars=['SDR','SIR'], var_name='Metric')

    ## Violin Plot
    #sns.set(style="whitegrid", context="paper", palette="pastel", color_codes=True, font_scale=0.9)
    #plt.figure(figsize=(3.38649, 3.38649))
    pal = sns.cubehelix_palette(7)
    pal = sns.cubehelix_palette(8, start=.5, rot=-.75)
    bicolor_pal = [pal[2], pal[5]]
    scaling = 0.66
    sns.set(style='whitegrid', context='paper', 
            #palette=sns.light_palette('navy', n_colors=2),
            palette=bicolor_pal,
            font_scale=0.9,
            rc={
                'figure.figsize':(scaling * 3.38649,scaling * 3.338649), 
                'lines.linewidth':0.5,
                #'font.family': u'Roboto',
                #'font.sans-serif': [u'Roboto Bold'],
                #'text.usetex': False,
                }
            )

    vps = (3.38649 * scaling , 3.338649 * scaling)

    # reverse order or 'learn' and 'anechoic'
    order_index = index_str.copy()
    order_index[:2] = order_index[1::-1]

    g1 = sns.factorplot(x="n_echoes", y="value",
        hue="Speaker gender", row="Metric",
        data=mdf, kind="violin", split=True,
        scale='area', #palette={'Male':'b', 'Female':'r'},
        order=order_index, sharey=False, legend=False,
        size=vps[1] / 2, aspect=vps[0]/vps[1] * 2)

    g1.set_titles('')

    leg = g1.axes[0][0].legend(framealpha=0.6, frameon=True, loc='upper left', fontsize='xx-small')
    leg.get_frame().set_linewidth(0)

    ax1 = g1.axes.flat[0]
    ax1.set_ylim([-2.5,7.5])
    ax1.set_ylabel('SDR')
    ax1.set_yticks([0., 2., 4., 6.])

    ax2 = g1.axes.flat[1]
    ax2.set_ylim([-2.5,13.5])
    ax2.set_ylabel('SIR')
    ax2.set_yticks([0., 3., 6., 9., 12.])
    for ax in g1.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.1,0.5)
    g1.set_xlabels('Number of echoes')
    g1.set_xticklabels(rotation=60, fontsize='xx-small')
    for ax in g1.axes[:,0]:
        ax.get_xaxis().set_label_coords(0.61, -0.30)

    sns.despine(left=True)

    plt.tight_layout(pad=0.5)

    plt.savefig(plot_basename + '_violin_plot.pdf')

    ## Facetgrid
    '''
    sns.set(style='white', context='paper', palette='pastel', font_scale=0.9,
            rc={
                'axes.facecolor': (0, 0, 0, 0),
                'figure.figsize':(3.38649,3.338649), 
                'lines.linewidth':1.,
                'font.family': u'Roboto',
                'font.sans-serif': [u'Roboto Bold'],
                'text.usetex': False,
                })
    pal = sns.cubehelix_palette(len(index_str), rot=-.25, light=.7)

    g = sns.FacetGrid(mdf, row="n_echoes", col='Metric', hue="n_echoes", 
            aspect= (vps[0])/(vps[1] / len(index_str)) / 2, size=vps[1]/len(index_str), 
            palette=pal, row_order=index_str,
            sharex=False, sharey=False)
    g.map(sns.kdeplot, "value", clip_on=True, shade=True, alpha=1, lw=1, bw=.2)
    g.map(sns.kdeplot, "value", clip_on=True, color="w", lw=0.2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=True)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(1., .2, label, fontweight="bold", color=color, 
                ha="right", va="center", transform=ax.transAxes)

    g.map(label, 'n_echoes')

    g.axes[-1,0].set_xlabel('SDR')
    for ax in g.axes[:,0]:
        ax.set_xlim([-2.,7.5])
        ax.set_ylim([0, 0.8])
    for ax in g.axes[:-1,0]:
        ax.set_xticks([])
    g.axes[-1,0].set_xticks([0., 2., 4., 6.])

    g.axes[-1,1].set_xlabel('SIR')
    for ax in g.axes[:,1]:
        ax.set_xlim([-2.5,13.5])
        ax.set_ylim([0, 0.5])
    for ax in g.axes[:-1,1]:
        ax.set_xticks([])
    g.axes[-1,1].set_xticks([0., 3., 6., 9., 12.])

    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.tight_layout()
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.5)
    '''

    if plot_flag:
        plt.show()
