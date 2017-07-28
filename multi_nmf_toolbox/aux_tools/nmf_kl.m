function [W, H, E] = nmf_kl(V, W0, H0, iter_num, q_noise_var)

%
% [W, H, E] = nmf_kl(V, W0, H0, iter_num, q_noise_var);
%
% NMF decomposition using multiplicative update (MU) rules and
% Kullback-Leibler (KL) divergence
%
%
% input 
% -----
%
% V                 : nonnegative data matrix to factorize (V = W*H) [F x N]
% W0                : initial matrix of bases [F x K]
% H0                : initial matrix of contributions [K x N]
% iter_num          : (opt) number of EM iterations (def = 100)
% q_noise_var       : (opt) quantization noise variance (def = eps)
%
% output
% ------
%
% W                 : estimated matrix of bases [F x K]
% H                 : estimated matrix of contributions [K x N]
% E                 : vector of approximation errors (per iteration)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010 Alexey Ozerov
% (alexey.ozerov -at- irisa.fr)
%
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eps = 2.2204e-32;                                       %very small constant

if nargin < 4 || isempty(iter_num)
    iter_num = 100;
end;

if nargin < 5 || isempty(q_noise_var)
    q_noise_var = eps;
end;

[F,N]=size(V);
K = size(W0,2);

W = W0;
H = H0;

S=sum(V(:));
O=ones(F,N);
o=ones(1,K);
WH=W*H+q_noise_var;

E = zeros(1, iter_num);

% MAIN LOOP
for iter = 1:iter_num
    % Update of W & H:
    W = W.*((V./(WH+eps))*H')./(O*H'+eps);            % Update of W
    w=sum(W,1); d=diag(o./w); W=W*d;                  % Normalization of columns of W
    d=diag(w); H=d*H;                                 % Energy transfer to H
    WH=W*H + q_noise_var;
    H = H.*(W'*(V./WH))./(W'*O+eps);                  % Udpate of H

    % Approximation error with KL divergence:
    WH=W*H + q_noise_var;
    E(iter) = sum(sum(V.*log((V./WH)+eps)-V+WH))/S;         % normalized approximation error

    fprintf('MU-KL-NMF: iteration %d of %d, approximation error = %f\n', iter, iter_num, E(iter));
end;
