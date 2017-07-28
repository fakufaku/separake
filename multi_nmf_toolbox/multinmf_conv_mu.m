function [Q, W, H, cost] = multinmf_conv_mu(V, n_iter, Q, W, H, part, switch_Q, switch_W, switch_H)

% Multichannel NMF minimizing Itakura-Saito divergence through multiplicative updates
%
% [Q, W, H, cost] = multinmf_conv_mu(V, n_iter, Q, W, H, part, switch_Q, switch_W, switch_H)
%
% Input:
%   - V: positive matrix data       (F x N x n_c)
%   - n_iter: number of iterations
%   - init_Q: squared filters       (F x n_c x n_s)
%   - init_W: basis                 (F x K)
%   - init_H: activation coef.      (K x N)
%   - part : component indices
%   - switch_W, switch_H, switch_Q: (opt) switches (0 or 1) (def = 1)
%
% Output:
%   - Estimated Q, W and H
%   - Cost through iterations betw. data power and fitted variance.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010 Cedric Fevotte
% (cedric.fevotte -at- telecom-paristech.fr)
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

% Stable version (11/09/08); scales fixed properly.
% This version updates V_ap after each parameter update, leading to faster
% convergence

if nargin < 7 || isempty(switch_Q)
    switch_Q = 1;
end;

if nargin < 8 || isempty(switch_W)
    switch_W = 1;
end;

if nargin < 9 || isempty(switch_H)
    switch_H = 1;
end;

[F,N,n_c] = size(V);
n_s = size(Q,3);

%% Definitions %%
V_ap = zeros(F,N,n_c);
cost = zeros(1,n_iter);

%% Compute app. variance structure V_ap %%
for j=1:n_s
    P_j = W(:,part{j}) * H(part{j},:);    
    for i=1:n_c
        V_ap(:,:,i) = V_ap(:,:,i) + repmat(Q(:,i,j),1,N) .* P_j;
    end
end

cost(1) = sum(V(:)./V_ap(:) - log(V(:)./V_ap(:))) - F*N*n_c;

for iter = 2:n_iter
    %%% Update Q %%%
    if switch_Q
        for j=1:n_s
            P_j = W(:,part{j}) * H(part{j},:);
            for i=1:n_c
                Q_old = Q(:,i,j);
                Q(:,i,j) = Q(:,i,j) .* (sum(V_ap(:,:,i).^-2 .* P_j.*V(:,:,i),2) ./ sum(V_ap(:,:,i).^-1 .* P_j,2));
                V_ap(:,:,i) = V_ap(:,:,i) + repmat(Q(:,i,j)-Q_old,1,N) .* P_j;
            end
        end
    end

    %%% Update W %%%
    if switch_W
        for j=1:n_s
            Kj   = length(part{j});
            Wnum = zeros(F,Kj);
            Wden = zeros(F,Kj);
            for i=1:n_c
                Wnum = Wnum + repmat(Q(:,i,j),1,Kj) .* ((V_ap(:,:,i).^-2 .* V(:,:,i)) * H(part{j},:)');
                Wden = Wden + repmat(Q(:,i,j),1,Kj) .* (V_ap(:,:,i).^-1 * H(part{j},:)');
            end

            Wj_old = W(:,part{j});
            W(:,part{j}) = W(:,part{j}) .* (Wnum./Wden);

            for i=1:n_c
                V_ap(:,:,i) = V_ap(:,:,i) + repmat(Q(:,i,j),1,N) .* ((W(:,part{j})-Wj_old)*H(part{j},:));
            end

        end

    end

    %%% Update H %%%
    if switch_H
        for j=1:n_s
            Kj   = length(part{j});
            Hnum = zeros(Kj,N);
            Hden = zeros(Kj,N);
            for i=1:n_c
                tmpH = (repmat(Q(:,i,j),1,Kj).*W(:,part{j}))';
                Hnum = Hnum + tmpH * (V_ap(:,:,i).^-2 .* V(:,:,i));
                Hden = Hden + tmpH * V_ap(:,:,i).^-1;
            end

            Hj_old = H(part{j},:);
            H(part{j},:) = H(part{j},:) .* (Hnum./Hden);

            for i=1:n_c
                V_ap(:,:,i) = V_ap(:,:,i) + repmat(Q(:,i,j),1,N) .* (W(:,part{j})*(H(part{j},:)-Hj_old));
            end

        end

    end

    cost(iter) = sum(V(:)./V_ap(:) - log(V(:)./V_ap(:))) - F*N*n_c;

    %%% Normalize %%%

    %% Scale Q / W %%
    if switch_Q && switch_W
        Q = permute(Q,[2 1 3]);
        Q = Q(:,:);
        scale = sum(Q,1);
        Q = Q ./ repmat(scale,n_c,1);
        Q = reshape(Q,n_c,F,n_s);
        Q = permute(Q,[2 1 3]);
        for j=1:n_s
            scale_j = scale((j-1)*F+1:j*F)';
            W(:,part{j}) = W(:,part{j}) .* repmat(scale_j,1,length(part{j}));
        end
    end

    %% Scale W / H %%
    if switch_W && switch_H
        scale = sum(W,1);
        W = W ./ repmat(scale ,F,1);
        H = H .* repmat(scale',1,N);
    end

    fprintf('MU update: iteration %d of %d, cost = %f\n', iter, n_iter, cost(iter));
end


% %%% Equivalent normalization for A %%%
%
% %% Scale A / W %%
% if switch_A && switch_W
%     A = permute(Q,[2 1 3]);
%     A = Q(:,:);
%     scale = sum(abs(A).^2,1);
%     phase = angle(A(1,:));
%     A = A ./ repmat(sqrt(scale) .* exp(j*phase),n_c,1);
%     A = reshape(A,n_c,F,n_s);
%     A = permute(A,[2 1 3]);
%        for j=1:n_s
%            scale_j = scale((j-1)*F+1:j*F)';
%            W(:,part{j}) = W(:,part{j}) .* repmat(scale_j,1,length(part{j}));
%            %P(:,:,j) = P(:,:,j) .* repmat(scale_j,1,N);
%        end
% end
%
% %% Scale W / H %%
% if switch_W && switch_H
%     scale = sum(W(:,part{j}),1);
%     W(:,part{j}) = W(:,part{j}) ./ repmat(scale ,F,1);
%     H(part{j},:) = H(part{j},:) .* repmat(scale',1,N);
% end
