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
if ~exist('estmix_conv_5cm.p','file')
    error('Please download estmix_conv_5cm.p from http://sisec.wiki.irisa.fr/tiki-index.php?page=Under-determined+speech+and+music+mixtures (see README.txt)');
end;

% Input time-frequency representation
fprintf('Input time-frequency representation\n');
[x, fs]=wavread([data_dir file_prefix '_mix.wav']);
x = x.';
mix_nsamp = size(x,2);
X=stft_multi(x,stft_win_len);

fprintf('INITIALIZATION (using convolutive filters estimation and source separation via binary masking)\n\n');

% Estimation of the frequency-dependent mixing matrix
fprintf('Estimation of the frequency-dependent mixing matrix\n');
wlen=2*(size(X,1)-1);
Ae=estmix_conv_5cm([X;conj(X(wlen/2:-1:2,:,:))],nsrc);
Ae=Ae(:,:,1:wlen/2+1);

% Source separation via binary masking
fprintf('Source separation via binary masking\n');
Se_bm=sep_binmask(X,Ae);

% Computation of the spatial source images
fprintf('Computation of the spatial source images\n');
Ie_bm=src_image(Se_bm,Ae);
ie_bm=istft_multi(Ie_bm,mix_nsamp);
for j=1:nsrc,
    wavwrite(reshape(ie_bm(j,:,:),mix_nsamp,2),fs,[results_dir file_prefix '_sim_init_bm_' int2str(j) '.wav']);
end

fprintf('Source separation via multichannel NMF EM algorithm\n\n');

fprintf('Parameters initialization\n\n');

A_init = Ae;

% compute ISTFT and STFT (it is not the same, since STFT is a redundant representation)
Se_bm_rec = stft_multi(istft_multi(Se_bm,mix_nsamp),stft_win_len);

% initialize W and H from bm separated sources
[W_init, H_init, source_NMF_ind] = Init_KL_NMF_fr_sep_sources(Se_bm_rec, NMF_CompPerSrcNum);

% initialize additive noise variances as mixture PSD / 100 
mix_psd = 0.5 * (mean(abs(X(:,:,1)).^2 + abs(X(:,:,2)).^2, 2));
Sigma_b_init = mix_psd / 100;

% run 200 iterations of multichannel NMF EM algorithm (with annealing and noise injection)
A_init = permute(A_init, [3 1 2]);

[W_EM, H_EM, Ae_EM, Sigma_b_EM, Se_EM, log_like_arr] = ...
    multinmf_conv_em(X, W_init, H_init, A_init, Sigma_b_init, source_NMF_ind, 200, 2);

Ae_EM = permute(Ae_EM, [2 3 1]);

% Computation of the spatial source images
fprintf('Computation of the spatial source images\n');
Ie_EM=src_image(Se_EM,Ae_EM);
ie_EM=istft_multi(Ie_EM,mix_nsamp);
for j=1:nsrc,
    wavwrite(reshape(ie_EM(j,:,:),mix_nsamp,2),fs,[results_dir file_prefix '_sim_EM_' int2str(j) '.wav']);
end

% Evaluation of the estimated source images (binary masking used for initialization of EM)
fprintf('Evaluation of the source images via binary masking (used for initialization of EM):\n');
i=zeros(nsrc,mix_nsamp,2);
for j=1:nsrc,
    i(j,:,:)=reshape(wavread([data_dir file_prefix '_sim_' int2str(j) '.wav']),1,mix_nsamp,2);
end
[SDRi,ISRi,SIRi,SARi,permi]=bss_eval_images(ie_bm,i)

% Evaluation of the estimated source images (multichannel NMF EM algorithm)
fprintf('Evaluation of the source images estimated via multichannel NMF EM algorithm:\n');
i=zeros(nsrc,mix_nsamp,2);
for j=1:nsrc,
    i(j,:,:)=reshape(wavread([data_dir file_prefix '_sim_' int2str(j) '.wav']),1,mix_nsamp,2);
end
[SDRi,ISRi,SIRi,SARi,permi]=bss_eval_images(ie_EM,i)
