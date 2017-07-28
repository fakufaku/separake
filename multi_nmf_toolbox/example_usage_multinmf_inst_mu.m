function example_usage_multinmf_inst_mu()

%
% example_usage_multinmf_inst_mu();
%
% Example of usage of MU rules for multichannel NMF decomposition in
%   linear instantaneous mixture
%
%
% input 
% -----
%
% ...
%
% output
% ------
%
% estimated source images are written in the results_dir
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010 Alexey Ozerov
% (alexey.ozerov -at- irisa.fr)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%
% If you use this code please cite this paper
%
% A. Ozerov and C. Fevotte,
% "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation,"
% IEEE Trans. on Audio, Speech and Lang. Proc. special issue on Signal Models and Representations
% of Musical and Environmental Sounds, vol. 18, no. 3, pp. 550-563, March 2010.
% Available: http://www.irisa.fr/metiss/ozerov/Publications/OzerovFevotte_IEEE_TASLP10.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


NMF_CompPerSrcNum = 4;
nsrc = 3;

stft_win_len = 2048;

data_dir = 'data/Shannonsongs/';
results_dir = 'data/Shannonsongs/';
file_prefix = 'Shannonsongs_Sunrise_inst_sh';

addpath('aux_tools');

% Input time-frequency representation
fprintf('Input time-frequency representation\n');
[x, fs]=wavread([data_dir file_prefix '_mix.wav']);
x = x.';
mix_nsamp = size(x,2);
X=stft_multi(x,stft_win_len);

nbin = size(X,1);
nfram = size(X,2);

% Random initialization of multichannel NMF parameters
fprintf('Random initialization of multichannel NMF parameters\n');
K = NMF_CompPerSrcNum * nsrc;
source_NMF_ind = cell(1,nsrc);
for j = 1:nsrc
    source_NMF_ind{j} = [1:NMF_CompPerSrcNum] + (j-1)*NMF_CompPerSrcNum;
end;
mix_psd = 0.5 * (mean(abs(X(:,:,1)).^2 + abs(X(:,:,2)).^2, 2));
A_init = 0.5 * (1.9 * abs(randn(2, nsrc)) + 0.1 * ones(2, nsrc));
% W is intialized so that its enegy follows mixture PSD
W_init = 0.5 * (abs(randn(nbin,K)) + ones(nbin,K)) .* (mix_psd * ones(1,K));
H_init = 0.5 * (abs(randn(K,nfram)) + ones(K,nfram));



% run 500 iterations of multichannel NMF MU rules
Q_init = abs(A_init).^2;

[Q_MU, W_MU, H_MU, cost] = multinmf_inst_mu(abs(X).^2, 500, Q_init, W_init, H_init, source_NMF_ind);


% Reconstruction of the spatial source images
fprintf('Reconstruction of the spatial source images\n');
Q_MU_conv = zeros(nbin, 2, nsrc);
for f = 1:nbin
    Q_MU_conv(f,:,:) = Q_MU;
end;
Ie_MU = multinmf_recons_im(X, Q_MU_conv, W_MU, H_MU, source_NMF_ind);
ie_MU=istft_multi(Ie_MU,mix_nsamp);
for j=1:nsrc,
    wavwrite(reshape(ie_MU(j,:,:),mix_nsamp,2),fs,[results_dir file_prefix '_sim_MU_' int2str(j) '.wav']);
end

% Plot estimated W and H
fprintf('Plot estimated W and H\n');
plot_ind = 1;
for k = 1:NMF_CompPerSrcNum
    for j = 1:nsrc
        subplot(NMF_CompPerSrcNum, nsrc, plot_ind);
        plot(max(log10(max(W_MU(:,source_NMF_ind{j}(k)), 1e-40)), -10));
        title(sprintf('Source_%d, log10(W_%d)', j, k));
        plot_ind = plot_ind + 1;
    end;
end;
figure;
plot_ind = 1;
for k = 1:NMF_CompPerSrcNum
    for j = 1:nsrc
        subplot(NMF_CompPerSrcNum, nsrc, plot_ind);
        plot(H_MU(source_NMF_ind{j}(k),:));
        title(sprintf('Source_%d, H_%d', j, k));
        plot_ind = plot_ind + 1;
    end;
end;
