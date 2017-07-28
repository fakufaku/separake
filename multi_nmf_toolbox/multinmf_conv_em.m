function [W,H,A,Sigma_b,S,log_like_arr] = ...
    multinmf_conv_em(X, W0, H0, A0, Sigma_b0, source_NMF_ind, iter_num, SimAnneal_flag, Sigma_b_Upd_flag)

%
% [W,H,A,Sigma_b,S,log_like_arr] = ...
%    multinmf_conv_em(X, W0, H0, A0, Sigma_b0, source_NMF_ind, iter_num, SimAnneal_flag, Sigma_b_Upd_flag);
%
% EM algorithm for multichannel NMF decomposition in convolutive mixture (stereo)
%
%
% input 
% -----
%
% X                 : truncated STFT of multichannal mixture [F x N x 2]
% W0                : initial matrix of bases [F x K]
% H0                : initial matrix of contributions [K x N]
% A0                : initial (complex-valued) convolutive mixing matrices [F x 2 x J]
% Sigma_b0          : initial additive noise covariances [F x 1] vector
% source_NMF_ind    : source indices in the NMF decomposition
% iter_num          : (opt) number of EM iterations (def = 100)
% SimAnneal_flag    : (opt) simulated annealing flag (def = 1)
%                         0 = no annealing
%                         1 = annealing
%                         2 = annealing with noise injection
% Sigma_b_Upd_flag  : (opt) Sigma_b update flag (def = 0)
%
% output
% ------
%
% W                 : estimated matrix of bases [F x K]
% H                 : estimated matrix of contributions [K x N]
% A                 : estimated (complex-valued) convolutive mixing matrices [F x 2 x J]
% Sigma_b           : final additive noise covariances [F x 1] vector
% S                 : estimated truncated STFTs of sources [F x N x J]
% log_like_arr      : array of log-likelihoods
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

% some constants
final_ann_noise_var = 3e-11;       % final annealing noise variance

if nargin < 7 || isempty(iter_num)
    iter_num = 100;
end;

if nargin < 8 || isempty(SimAnneal_flag)
    SimAnneal_flag = 1;
end;

if nargin < 9 || isempty(Sigma_b_Upd_flag)
    Sigma_b_Upd_flag = 0;
end;

[F, N, I] = size(X);
K = size(W0, 2);
J = length(source_NMF_ind);

if I ~= 2
    error('Multi_NMF_EM_conv: number of channels must be 2');
end;

if SimAnneal_flag && Sigma_b_Upd_flag
    error('The flags SimAnneal_flag and Sigma_b_Upd_flag cannot be on simultaneously');
end;

Xb = X;

W = W0;
H = H0;
A = A0;
Sigma_b = Sigma_b0;

O = ones(1,N);

sigma_ss = zeros(F,N,J);

Sigma_x = zeros(F,N,2,2);
Inv_Sigma_x = zeros(F,N,2,2);

Gs = zeros(F,N,J,2);
Gs_x = zeros(F,N,J);

bar_Rxs = zeros(F,2,J);
bar_Rss = zeros(F,J,J);
bar_Rxx = zeros(F,2,2);

bar_A = zeros(F,2,K);

Vc = zeros(F,N,K);

log_like_arr = zeros(1,iter_num);

% initialize simulated annealing variances (if necessary)
if SimAnneal_flag
    Sigma_b_anneal = zeros(F,iter_num);
    for iter = 1:iter_num
        Sigma_b_anneal(:,iter) = ((sqrt(Sigma_b0) * (iter_num - iter) + ones(F, 1) * sqrt(final_ann_noise_var) * iter) / iter_num).^2;
    end;
end;

% MAIN LOOP
for iter = 1:iter_num
    fprintf('EM iteration %d of %d\n', iter, iter_num);
    
    % store parameters estimated on previous iteration
    W_prev       = W;
    H_prev       = H;
    A_prev       = A;
    Sigma_b_prev = Sigma_b;

    % E-step: compute expectations of natural suffitient statistics
    fprintf('   E-step\n');

    % compute a priori source variances
    sigma_ss = sigma_ss * 0;
    for j = 1:J
        for k = 1:length(source_NMF_ind{j})
            sigma_ss(:,:,j) = sigma_ss(:,:,j) + W(:,source_NMF_ind{j}(k))*H(source_NMF_ind{j}(k),:);
        end;
    end;

    if SimAnneal_flag
        Sigma_b = Sigma_b_anneal(:,iter);
        
        if SimAnneal_flag == 2   % with noise injection
            Noise = randn(F,N,2) + sqrt(-1) * randn(F,N,2);
            Noise(:,:,1) = Noise(:,:,1) .* (sqrt(Sigma_b / 2) * ones(1,N));
            Noise(:,:,2) = Noise(:,:,2) .* (sqrt(Sigma_b / 2) * ones(1,N));

            Xb = X + Noise;
        end;
    end;
    
    % compute the Sigma_x matrix
    Sigma_x(:,:,1,1) = Sigma_b * O;
    Sigma_x(:,:,1,2) = 0;
    Sigma_x(:,:,2,1) = 0;
    Sigma_x(:,:,2,2) = Sigma_b * O;
    for j = 1:J
        Sigma_x(:,:,1,1) = Sigma_x(:,:,1,1) + (abs(A(:,1,j)).^2 * O) .* sigma_ss(:,:,j);
        Sigma_x(:,:,1,2) = Sigma_x(:,:,1,2) + ((A(:,1,j) .* conj(A(:,2,j))) * O) .* sigma_ss(:,:,j);
        Sigma_x(:,:,2,1) = conj(Sigma_x(:,:,1,2));
        Sigma_x(:,:,2,2) = Sigma_x(:,:,2,2) + (abs(A(:,2,j)).^2 * O) .* sigma_ss(:,:,j);
    end;

    % compute the inverse of Sigma_x matrix
    Det_Sigma_x = Sigma_x(:,:,1,1) .* Sigma_x(:,:,2,2) - abs(Sigma_x(:,:,1,2)).^2;
    Inv_Sigma_x(:,:,1,1) =   Sigma_x(:,:,2,2) ./ Det_Sigma_x;
    Inv_Sigma_x(:,:,1,2) = - Sigma_x(:,:,1,2) ./ Det_Sigma_x;
    Inv_Sigma_x(:,:,2,1) =   conj(Inv_Sigma_x(:,:,1,2));
    Inv_Sigma_x(:,:,2,2) =   Sigma_x(:,:,1,1) ./ Det_Sigma_x;
        
    % compute log-likelihood
    log_like = -sum(sum(log(Det_Sigma_x * pi) + ...
        Inv_Sigma_x(:,:,1,1) .* (abs(Xb(:,:,1)).^2) + ...
        Inv_Sigma_x(:,:,2,2) .* (abs(Xb(:,:,2)).^2) + ...
        2 * real(Inv_Sigma_x(:,:,1,2) .* conj(Xb(:,:,1)) .* Xb(:,:,2)) )) / (N * F);
    if iter > 1
        log_like_diff = log_like - log_like_arr(iter-1);
        fprintf('Log-likelihood: %f   Log-likelihood improvement: %f\n', log_like, log_like_diff);
    else
        fprintf('Log-likelihood: %f\n', log_like);
    end;
    log_like_arr(iter) = log_like;
    
    for j = 1:J
        % compute S-Wiener gain
        Gs(:,:,j,1) = ((conj(A(:,1,j)) * O) .* Inv_Sigma_x(:,:,1,1) + ...
                       (conj(A(:,2,j)) * O) .* Inv_Sigma_x(:,:,2,1)) .* sigma_ss(:,:,j);
        Gs(:,:,j,2) = ((conj(A(:,1,j)) * O) .* Inv_Sigma_x(:,:,1,2) + ...
                       (conj(A(:,2,j)) * O) .* Inv_Sigma_x(:,:,2,2)) .* sigma_ss(:,:,j);

        % compute Gs_x
        Gs_x(:,:,j) = Gs(:,:,j,1) .* Xb(:,:,1) + Gs(:,:,j,2) .* Xb(:,:,2);

        % compute average Rxs
        bar_Rxs(:,1,j) = mean(Xb(:,:,1) .* conj(Gs_x(:,:,j)), 2);
        bar_Rxs(:,2,j) = mean(Xb(:,:,2) .* conj(Gs_x(:,:,j)), 2);
    end;
    
    for j1 = 1:J
        % compute average Rss
        for j2 = 1:J
            bar_Rss(:,j1,j2) = mean(Gs_x(:,:,j1) .* conj(Gs_x(:,:,j2)) ...
                - (Gs(:,:,j1,1) .* (A(:,1,j2) * O) + Gs(:,:,j1,2) .* (A(:,2,j2) * O)) .* sigma_ss(:,:,j2), 2);
        end;
        bar_Rss(:,j1,j1) = bar_Rss(:,j1,j1) + mean(sigma_ss(:,:,j1), 2);
    end;

    % compute average Rxx
    bar_Rxx(:,1,1) = mean(abs(Xb(:,:,1)).^2, 2);
    bar_Rxx(:,2,2) = mean(abs(Xb(:,:,2)).^2, 2);
    bar_Rxx(:,1,2) = mean(Xb(:,:,1) .* conj(Xb(:,:,2)), 2);
    bar_Rxx(:,2,1) = conj(bar_Rxx(:,1,2));
    
    % TO ASSURE that Rss = Rss'
    for f = 1:F
        bar_Rss(f,:,:) = (squeeze(bar_Rss(f,:,:)) + squeeze(bar_Rss(f,:,:))') / 2;
    end;

    % compute extended mixing matrix A
    for j = 1:J
        for k = source_NMF_ind{j}
            bar_A(:,:,k) = A(:,:,j);
        end;
    end;

    for k = 1:K
        % compute a priori component variances
    	sigma_cc_k = W(:,k)*H(k,:);

        % compute C-Wiener gain
        Gc_k_1 = ((conj(bar_A(:,1,k)) * O) .* Inv_Sigma_x(:,:,1,1) + ...
                  (conj(bar_A(:,2,k)) * O) .* Inv_Sigma_x(:,:,2,1)) .* sigma_cc_k;
        Gc_k_2 = ((conj(bar_A(:,1,k)) * O) .* Inv_Sigma_x(:,:,1,2) + ...
                  (conj(bar_A(:,2,k)) * O) .* Inv_Sigma_x(:,:,2,2)) .* sigma_cc_k;

        % compute Gc_x
        Gc_x_k = Gc_k_1 .* Xb(:,:,1) + Gc_k_2 .* Xb(:,:,2);

        % compute components suffitient natural statistics
        % IT IS IMPORTANT TO TAKE A REAL PART !!!!
        Vc(:,:,k) = abs(Gc_x_k).^2 + sigma_cc_k ...
            - real(Gc_k_1 .* (bar_A(:,1,k) * O) + Gc_k_2 .* (bar_A(:,2,k) * O)) .* sigma_cc_k;
    end;
    
    % M-step: re-estimate parameters
    fprintf('   M-step\n');

    % re-estimate A
    for f = 1:F
        A(f,:,:) = squeeze(mean(bar_Rxs(f,:,:), 1)) * inv(squeeze(mean(bar_Rss(f,:,:), 1)));
    end;
        
    % re-estimate noise variances (if necessary)
    if Sigma_b_Upd_flag
        for f = 1:F
            Sigma_b(f) = 0.5 * real(trace(squeeze(bar_Rxx(f,:,:)) - ...
                squeeze(A(f,:,:)) * squeeze(bar_Rxs(f,:,:))' - ...
                squeeze(bar_Rxs(f,:,:)) * squeeze(A(f,:,:))' + ...
                squeeze(A(f,:,:)) * squeeze(bar_Rss(f,:,:)) * squeeze(A(f,:,:))'));
        end;
    end;

    % re-estimate W, and then H
    for k=1:K
        W(:,k) = sum(Vc(:,:,k) ./ (ones(F,1)*H(k,:)), 2) / N;
        H(k,:) = sum(Vc(:,:,k) ./ (W(:,k)*ones(1,N)), 1) / F;
    end;
    
    % Normalization
    for j = 1:J
        nonzero_f_ind = find(A(:, 1, j) ~= 0);
        A(nonzero_f_ind, 2, j) = A(nonzero_f_ind, 2, j) ./ sign(A(nonzero_f_ind, 1, j));
        A(nonzero_f_ind, 1, j) = A(nonzero_f_ind, 1, j) ./ sign(A(nonzero_f_ind, 1, j));
        
        A_scale = abs(A(:, 1, j)).^2 + abs(A(:, 2, j)).^2;
        A(:, :, j) = A(:, :, j) ./ (sqrt(A_scale) * [1, 1]);   % Normalisation of mixing filters A
        W(:,source_NMF_ind{j}) = W(:,source_NMF_ind{j}) .* ...
            (A_scale * ones(1,length(source_NMF_ind{j})));     % Energy transfer to W components
    end;
    w=sum(W,1); d=diag(ones(1,K)./w); W=W*d;                   % Normalisation of W components
    d=diag(w); H=d*H;                                          % Energy transfer to H
end;

% source estimates
S = Gs_x;

% return parameters estimated on previous iteration, since they were used
% for source estimates computation
W       = W_prev;
H       = H_prev;
A       = A_prev;
Sigma_b = Sigma_b_prev;
