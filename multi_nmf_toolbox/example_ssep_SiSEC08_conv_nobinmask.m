function example_ssep_SiSEC08_conv()

%
% example_ssep_SiSEC08_conv();
%
% Multichannel NMF EM algorithm for SiSEC 2008 evaluation campaign (http://sisec.wiki.irisa.fr/)
%   convolutive mixtures of "Under-determined speech and music mixtures" task
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

stft_win_len = 2048;

data_dir = 'data/SiSEC08/';
results_dir = 'data/SiSEC08/';
file_prefix = 'dev1_nodrums_synthconv_250ms_5cm';

addpath('aux_tools');

% manual counting of number of sources
nsrc = 0;
while exist(sprintf('%s%s_sim_%d.wav', data_dir, file_prefix, nsrc+1), 'file')
    nsrc = nsrc + 1;
end;
if nsrc < 2
    error('Number of sources must be at least 2');
end;

% Input time-frequency representation
fprintf('Input time-frequency representation\n');
[x, fs]=audioread([data_dir file_prefix '_mix.wav']);
x = x.';
mix_nsamp = size(x,2);
X=stft_multi(x,stft_win_len);

fprintf('Source separation via multichannel NMF EM algorithm\n\n');

fprintf('Parameters initialization\n\n');

nfram = size(X,2);
nbin = size(X,1);

% random initialization
mix_psd = 0.5 * (mean(abs(X(:,:,1)).^2 + abs(X(:,:,2)).^2, 2));
A_init = 0.5 * (1.9 * abs(randn(2, nsrc, nbin)) + 0.1 * ones(2, nsrc, nbin)) .* sign(randn(2, nsrc, nbin) + sqrt(-1)*randn(2, nsrc, nbin));

K = NMF_CompPerSrcNum * nsrc;
source_NMF_ind = cell(1,nsrc);
for j = 1:nsrc
    source_NMF_ind{j} = [1:NMF_CompPerSrcNum] + (j-1)*NMF_CompPerSrcNum;
end;

W_init = 0.5 * (abs(randn(nbin,K)) + ones(nbin,K)) .* (mix_psd * ones(1,K));
H_init = 0.5 * (abs(randn(K,nfram)) + ones(K,nfram));

% initialize additive noise variances as mixture PSD / 100 
Sigma_b_init = mix_psd / 100;

% run 200 iterations of multichannel NMF EM algorithm (with annealing and noise injection)
A_init = permute(A_init, [3 1 2]);

[W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr] = ...
    multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind, 20, 2);

Ae_EM = permute(Ae_EM, [2 3 1]);

% Computation of the spatial source images
fprintf('Computation of the spatial source images\n');
Ie_EM=src_image(Se_EM,Ae_EM);
ie_EM=istft_multi(Ie_EM,mix_nsamp);
for j=1:nsrc,
    audiowrite([results_dir file_prefix '_sim_EM_' int2str(j) '.wav'], reshape(ie_EM(j,:,:),mix_nsamp,2),fs);
end

% Evaluation of the estimated source images (multichannel NMF EM algorithm)
fprintf('Evaluation of the source images estimated via multichannel NMF EM algorithm:\n');
i=zeros(nsrc,mix_nsamp,2);
for j=1:nsrc,
    i(j,:,:)=reshape(audioread([data_dir file_prefix '_sim_' int2str(j) '.wav']),1,mix_nsamp,2);
end
[SDRi,ISRi,SIRi,SARi,permi]=bss_eval_images(ie_EM,i)
