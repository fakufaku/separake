import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from pyroomacoustics import median

if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description='Plot the medians')
    parser.add_argument('pickles', type=str, nargs='+', metavar='DIR',
            help='Pickle files containing all the medians to plot.')
    parser.add_argument('--with-confidence', action='store_true',
            help='Plot the confidence intervals for the median.')

    args = parser.parse_args()

    size = 1.65
    aspect = 3.38649 / size / 2

    pal = sns.cubehelix_palette(4, start=.5, rot=-.75)
    custom_pal = pal[1:]
    sns.set(style='whitegrid', context='paper', 
            palette=custom_pal, font_scale=0.9,
            rc={
                'figure.figsize':(3.38649,3.338649), 
                'lines.linewidth':1.,
                #'font.family': u'Roboto',
                #'font.sans-serif': [u'Roboto Bold'],
                'text.usetex': False,
                })

    df_median = []
    df_raw = []

    for pickle in args.pickles:
        df1 = pd.read_pickle(pickle)
        df2 = pd.read_pickle(pickle.replace('median','dataframe'))

        if '_mu_' in pickle:
            algo = 'MU-NMF'
        elif '_em_' in pickle:
            algo = 'EM-NMF'

        if 'SpkrDict' in pickle:
            dic = 'speak. dep.'
        elif 'UnivDict' in pickle:
            dic = 'univ.'

        label = algo + ', ' + dic

        df1['Algorithm'] = label
        df2['Algorithm'] = label

        df_median.append(df1.reset_index())
        df_raw.append(df2)

    df_median = pd.concat(df_median)
    df_raw = pd.concat(df_raw)

    index = np.sort(df_median.n_echoes.unique()).tolist()
    index_str = [str(i) for i in index]
    if -2 in index:
        index_str[index.index(-2)] = 'learn'
    if -1 in index:
        index_str[index.index(-1)] = 'anechoic'

    # Plot the median
    df = df_raw.replace({'n_echoes': dict(zip(index, index_str))})
    df = df.melt(
        id_vars=['n_echoes','Algorithm'], 
        value_vars=['SDR','SIR'], var_name='Metric')

    def med_ci(slice_values):
        ''' This is used to compute median and confidence interval in dataframe '''
        m, ci = median(np.array(slice_values['value']).flatten(), alpha=0.05)
        return pd.Series([m, ci[0], ci[1]], index=['value','ci_lo','ci_hi'])

    # compute median and confidence intervals
    df_median = df.groupby(['n_echoes','Algorithm','Metric'])
    df_median = df_median.apply(med_ci).reset_index()

    # reverse order or 'learn' and 'anechoic'
    order_index = index_str.copy()
    order_index[:2] = order_index[1::-1]

    g = sns.factorplot(data=df_median, x='n_echoes', y='value', hue='Algorithm', col='Metric',
            order=order_index, scale=0.8,
            size=size, aspect=aspect, legend=False, clip_on=True)

    if args.with_confidence:
        g = sns.FacetGrid(df_median, col="Metric", size=size, aspect=aspect)

    def func(x,y,h,lb,ub, **kwargs):
        ''' This used to plot assymteric error bars in factor plot '''
        data = kwargs.pop("data")
        color = kwargs.pop("color")
        ax = plt.gca()

        # from https://stackoverflow.com/a/37139647/4124317
        med = data.pivot(index=x, columns=h, values=y)
        errLo = data.pivot(index=x, columns=h, values=lb)
        errHi = data.pivot(index=x, columns=h, values=ub)

        for col in errLo:
            c = next(ax._get_lines.prop_cycler)['color']
            for ind, label in zip(index, index_str):
                xloc = [ind+2, ind+2]
                yloc = [med[col][label]+errLo[col][label], med[col][label]+errHi[col][label]]
                plt.plot(xloc, yloc, '-', color=c)

    #g.map_dataframe(func, 'n_echoes', 'value', 'Algorithm', 'ci_lo', 'ci_hi')

    g.set_titles('{col_name}')
    g.set_xlabels('Number of echoes')
    g.set_xticklabels(rotation=60, fontsize='xx-small')
    g.set(ylim=[0,8.3])
    g.set_ylabels('[dB]')
    leg = g.axes[0][0].legend(framealpha=0.6, frameon=True, loc='upper right',fontsize='xx-small')
    leg.get_frame().set_linewidth(0)
    for ax in g.axes.flat:
        #ax.get_xaxis().set_label_coords(0.5,-0.3)
        ax.get_xaxis().set_label_coords(0.61,-0.25)

    extreme = df_median[np.logical_and(df_median.n_echoes == 'anechoic', df_median.Algorithm == 'EM-NMF, speak. dep.')][['Metric','value']]

    g.axes[0][0].annotate('{:0.1f} dB'.format(float(extreme[extreme.Metric == 'SDR']['value'])), 
            xy=(0,8.1), xytext=(0,9.3),
            va="bottom", ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1., facecolor='black', shrinkA=0.5), 
            fontsize='xx-small', annotation_clip=False)
    g.axes[0][1].annotate('{:0.1f} dB'.format(float(extreme[extreme.Metric == 'SIR']['value'])), 
            xy=(0,8.1), xytext=(0,9.3),
            va="bottom", ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1., facecolor='black', shrinkA=0.5), 
            fontsize='xx-small', annotation_clip=False)

    g.despine(left=True)

    plt.tight_layout(pad=0.5)

    plt.savefig('figures/all_medians.pdf')

    # Plot the median with confidence intervals
    '''
    df = df_raw.replace({'n_echoes': dict(zip(index, index_str))})
    df = df.melt(
        id_vars=['n_echoes','Algorithm'], 
        value_vars=['SDR','SIR'], var_name='Metric')

    g = sns.factorplot(data=df, x='n_echoes', y='value', hue='Algorithm', col='Metric',
            order=index_str, estimator=np.median,
            size=size, aspect=aspect, legend=False, clip_on=True)
    g.set_titles('{col_name}')
    g.set_xlabels('Number of echoes')
    g.set_xticklabels(rotation=60, fontsize='xx-small')
    g.set(ylim=[0,8.3])
    g.set_ylabels('[dB]')
    leg = g.axes[0][0].legend(framealpha=0.6, frameon=True, loc='upper right',fontsize='xx-small')
    leg.get_frame().set_linewidth(0)
    for ax in g.axes.flat:
        ax.get_xaxis().set_label_coords(0.5,-0.3)
    g.axes[0][0].annotate('12.4 dB', xy=(1,8.1), xytext=(1,9.3),
            va="bottom", ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1., facecolor='black'), 
            fontsize='xx-small', annotation_clip=False)
    g.axes[0][1].annotate('11.2 dB', xy=(1,8.1), xytext=(1,9.3),
            va="bottom", ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1., facecolor='black'), 
            fontsize='xx-small', annotation_clip=False)

    def median_ci(n_echoes, algo, metric, **kwargs):
        ax = plt.gca()
        data = kwargs.pop('data')
        I = np.logical_and(data.n_echoes == n_echoes, data.Algorithm == algo)
        I = np.logical_and(I, data.Metric == metric)
        m, ci = median(data[I]['value'])
        plt.plot(n_echoes, ci, kind='bar')

    g.map_dataframe(median_ci, 'n_echoes', 'Algorithm', 'Metric')

    g.despine(left=True)

    plt.tight_layout()

    plt.savefig('figures/all_medians.pdf')
    '''

    plt.show()

