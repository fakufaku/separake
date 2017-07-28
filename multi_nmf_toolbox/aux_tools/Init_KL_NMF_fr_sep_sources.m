function [W, H, source_NMF_ind] = Init_KL_NMF_fr_sep_sources(S_sep, NMF_CompPerSrcNum)

%
% [W, H, source_NMF_ind] = Init_KL_NMF_fr_sep_sources(S_sep, NMF_CompPerSrcNum);
%
% Initialize NMF from STFTs of separated sources using Kullback-Leibler (KL) divergence
%
%
% input 
% -----
%
% S_sep             : half STFTs of separated sources [F x N x J]
% NMF_CompPerSrcNum : number of NMF components per source
%
% output
% ------
%
% W                 : estimated matrix of bases [F x K]
% H                 : estimated matrix of contributions [K x N]
% source_NMF_ind    : source indices in the NMF decomposition
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010 Alexey Ozerov
% (alexey.ozerov -at- irisa.fr)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% some constants
small_noise_var = 3e-11;       % small noise variance

[F, N, J] = size(S_sep);

K = 0;
source_NMF_ind = cell(1,J);
for j = 1:J
    source_NMF_ind{j} = (1:NMF_CompPerSrcNum) + (j-1) * NMF_CompPerSrcNum;
    K = K + length(source_NMF_ind{j});
end;

W = [];
H = [];

for j = 1:J
    V = abs(S_sep(:,:,j)).^2 + randn(F, N).^2 * small_noise_var;

    % run multiplicative algorithm with 10 iterations and 5 random
    % initializations, and keep the best initialization
    for init_ind = 1:5
        W_src = 0.3 + 1.7 * rand(F, length(source_NMF_ind{j})) * sqrt(mean(V(:)));
        H_src = 0.3 + 1.7 * rand(length(source_NMF_ind{j}), N) * sqrt(mean(V(:)));

        [W_src,H_src,E] = nmf_kl(V,W_src,H_src,10,small_noise_var);

        E_end = E(end);

        if init_ind > 1
            if E_end < E_end_best
                W_src_best = W_src;
                H_src_best = H_src;
                E_end_best = E_end;
            end;
        else
            W_src_best = W_src;
            H_src_best = H_src;        
            E_end_best = E_end;
        end;
    end;

    % now run 50 iterations multiplicative algorithm with best initialization
    [W_src, H_src] = nmf_kl(V,W_src_best,H_src_best,50,small_noise_var);

    W = [W, W_src];
    H = [H; H_src];
end;

W = max(W, small_noise_var * 100);
H = max(H, small_noise_var * 10);
