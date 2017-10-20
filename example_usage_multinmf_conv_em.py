import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

from multinmf_conv_em import multinmf_conv_em

def example_usage_multinmf_conv_em():
    #
    # example_usage_multinmf_conv_em()
    #
    # Example of usage of EM algorithm for multichannel NMF decomposition in
    #   convolutive mixture
    #
    #
    # input
    # -----
    #
    # ...
    #
    # output
    # ------
    #
    # estimated source images are written in the results_dir
    #
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Copyright 2017 Robin Scheibler, adapted to Python
    # Copyright 2010 Alexey Ozerov
    # (alexey.ozerov -at- irisa.fr)
    #
    # This software is distributed under the terms of the GNU Public License
    # version 3 (http://www.gnu.org/licenses/gpl.txt)
    #
    # If you use this code please cite this paper
    #
    # A. Ozerov and C. Fevotte,
    # "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
    # IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
    # of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
    # Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    NMF_CompPerSrcNum = 4
    nsrc = 3
    stft_win_len = 2048

    data_dir = 'data/Speech/'
    results_dir = 'data/Speech/'
    file_prefix = '3sources_3channels'

    # Input time-frequency representation
    print('Input time-frequency representation')
    fs, x = wavfile.read(data_dir + file_prefix + '_mix.wav')
    x = x / (2**15)
    mix_nsamp = x.shape[0]
    nchan = x.shape[1]


    # TODO STFT
    window = pra.cosine(stft_win_len)
    # X is (nchan, nframe, nbin)
    X = np.array(
            [pra.stft(x[:,ch], stft_win_len, stft_win_len // 2, win=window, transform=np.fft.rfft) for ch in range(nchan)]
            )
    # move axes to match Ozerov's order (nbin, nfram, nchan)
    X = np.moveaxis(X, [0,1,2], [2,1,0])
    nbin = X.shape[0]
    nfram = X.shape[1]


    # Random initialization of multichannel NMF parameters
    print('Random initialization of multichannel NMF parameters')
    K = NMF_CompPerSrcNum * nsrc
    source_NMF_ind = []
    for j in range(nsrc):
        source_NMF_ind.append(np.arange(NMF_CompPerSrcNum) + j * NMF_CompPerSrcNum)
    mix_psd = 0.5 * (np.mean(np.abs(X[:,:,0])**2 + np.abs(X[:,:,1])**2, axis=1))
    random_phases = random.randn(nchan, nsrc, nbin) + 1j * random.randn(nchan, nsrc, nbin)
    random_phases /= np.abs(random_phases)
    A_init = (0.5 *
            (1.9 * np.abs(random.randn(nchan, nsrc, nbin)) + 0.1 * np.ones((nchan, nsrc, nbin)))
            * random_phases
            )
    # W is intialized so that its enegy follows mixture PSD
    W_init = 0.5 * (
            ( np.abs(random.randn(nbin,K)) + np.ones((nbin,K)) )
            * ( mix_psd[:,np.newaxis] * np.ones((1,K)) )
            )
    H_init = 0.5 * ( np.abs(random.randn(K,nfram)) + np.ones((K,nfram)) )
    Sigma_b_init = mix_psd / 100
    print(Sigma_b_init.shape)

    # run 500 iterations of multichannel NMF EM algorithm (with annealing)
    A_init = np.moveaxis(A_init, [2], [0])

    # from scipy.io import loadmat
    # from_matlab = loadmat("/home/chutlhu/Documents/MATLAB/multi_nmf_toolbox/input.mat")
    # X = np.array(from_matlab["X"])
    # W_init = np.array(from_matlab["W_init"])
    # H_init = np.array(from_matlab["H_init"])
    # A_init = np.array(from_matlab["A_init"])
    # Sigma_b_init = np.array(from_matlab["Sigma_b_init"]).squeeze()
    # source_NMF_ind = from_matlab["source_NMF_ind"][0]
    # for idx, item in enumerate(source_NMF_ind):
    #  source_NMF_ind[idx] = item[0]-1

    W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr = \
        multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind, iter_num=1000)

    Ae_EM = np.moveaxis(Ae_EM, [0], [2])

    # Computation of the spatial source images
    print('Computation of the spatial source images\n')
    Ie_EM = np.zeros((nbin,nfram,nsrc,nchan), dtype=np.complex)
    for j in range(nsrc):
        for f in range(nbin):
            Ie_EM[f,:,j,:] = np.outer(Se_EM[f,:,j], Ae_EM[:,j,f])

    # Inverse STFT
    ie_EM = []
    for j in range(nsrc):
        # channel-wise istft with synthesis window
        ie_EM = []
        for ch in range(nchan):
            ie_EM.append(
                    pra.istft(Ie_EM[:,:,j,ch].T, stft_win_len, stft_win_len // 2, win=window, transform=np.fft.irfft)
                    )
        # write the separated source to a wav file
        out_filename = results_dir + '_sim_EM_' + str(j) + '.wav'
        wavfile.write(out_filename, fs, np.array(ie_EM).T)

    # Plot estimated W and H
    print('Plot estimated W and H')
    plt.figure()
    plot_ind = 1
    for k in range(NMF_CompPerSrcNum):
        for j in range(nsrc):
            plt.subplot(NMF_CompPerSrcNum, nsrc, plot_ind)
            plt.plot(np.log10(np.maximum(W_EM[:,source_NMF_ind[j][k]], 1e-40)))
            plt.title('Source_{}, log10(W_{})'.format(j, k))
            plot_ind += 1
    plt.tight_layout()

    plt.figure()
    plot_ind = 1
    for k in range(NMF_CompPerSrcNum):
        for j in range(nsrc):
            plt.subplot(NMF_CompPerSrcNum, nsrc, plot_ind)
            plt.plot(H_EM[source_NMF_ind[j][k],:])
            plt.title('Source_{}, H_{}'.format(j, k))
            plot_ind = plot_ind + 1
    plt.tight_layout()

    plt.show()

    plt.figure()
    plt.plot(log_like_arr)
    plt.show()

if __name__ == '__main__':
    example_usage_multinmf_conv_em()
