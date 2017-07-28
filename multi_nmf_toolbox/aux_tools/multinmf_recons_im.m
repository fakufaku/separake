function Im = multinmf_recons_im(X,Q,W,H,part)

%
% Reconstructs source images from multi-nmf factorization (conservative)
% 
% Im = multinmf_recons_im(X,Q,W,H,part)
% 
%
% Input:
%   - X: truncated STFT of multichannal mixture (F x N x n_c)
%   - Q: squared filters                        (F x n_c x n_s)
%   - W: basis                                  (F x K)
%   - H: activation coef.                       (K x N)
%   - part : component indices
%
% Output:
%   - Im : reconstructed source images (F x N x n_s x n_c)
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

[F,N,n_c] = size(X);
n_s = size(Q,3);

P = zeros(F,N,n_s);

for j=1:n_s
    P(:,:,j) = W(:,part{j}) * H(part{j},:);
end

Im = zeros(F, N, n_s, n_c);

if n_c == 2
    % Implementation for n_c > 2 cannot be easily vectorized as follows
    
    P1 = zeros(size(P));
    P2 = zeros(size(P));

    for j=1:n_s
        P1(:,:,j) = repmat(Q(:,1,j),1,N) .* P(:,:,j) ;
        P2(:,:,j) = repmat(Q(:,2,j),1,N) .* P(:,:,j) ;
    end

    for j=1:n_s
        Im(:,:,j,1) = (P1(:,:,j) ./ sum(P1, 3)) .* X(:,:,1);
        Im(:,:,j,2) = (P2(:,:,j) ./ sum(P2, 3)) .* X(:,:,2);
    end;
else
    error('multinmf_recons_im: number of channels must be 2');
end
