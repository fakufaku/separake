function S=sep_l0min(X,A)

% SEP_L0MIN STFT-domain source separation via lp-norm minimization with
% p->0 and with as many active sources as channels, given the mixing matrix
% for instantaneous mixtures or the frequency-dependent mixing matrix for
% convolutive mixtures
%
% S=sep_l0min(X,A)
%
% Inputs:
% X: nbin x nfram x nchan matrix containing mixture STFT coefficients with
% nbin frequency bins, nfram time frames and nchan channels
% A: either a nchan x nsrc mixing matrix (for instantaneous mixtures) or a
% nchan x nsrc x nbin frequency-dependent mixing matrix (for convolutive
% mixtures)
%
% Output:
% S: nbin x nfram x nsrc matrix containing the estimated source STFT
% coefficients
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2008 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Emmanuel Vincent, "Complex nonconvex lp norm minimization for
% underdetermined source separation," In Proc. Int. Conf. on Independent
% Component Analysis and Blind Source Separation (ICA), pp. 430-437, 2007.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors and warnings %%%
if nargin<2, error('Not enough input arguments.'); end
[nbin,nfram,nchan]=size(X);
if nbin==2*floor(nbin/2), error('The number of frequency bins must be odd.'); end
[nchan2,nsrc,nbin2]=size(A);
if nchan2~=nchan, error('The number of channels is different within the mixture and the mixing matrix.'); end
if nbin2==1,
    A=repmat(A,[1 1 nbin]);
elseif nbin2~=nbin,
    error('The number of frequency bins is different within the mixture and the frequency-dependent mixing matrix.');
end

%%% l0-norm minimization %%%
% Source activity patterns
act=zeros(2^nsrc,nsrc);
for j=1:nsrc,
    act(:,j)=reshape(repmat([0 1],[2^(nsrc-j) 2^(j-1)]),2^nsrc,1);
end
act=logical(act(sum(act,2)==nchan,:));
npat=size(act,1);
S=zeros(nbin,nfram,nsrc);
for f=1:nbin,
    % Source coefficients
    Xf=reshape(X(f,:,:),nfram,nchan).';
    SPf=zeros(nchan,nfram,npat);
    for p=1:npat,
        SPf(:,:,p)=inv(A(:,act(p,:)))*Xf;
    end
    % Selection of the best activity pattern
    [val,ind]=min(reshape(prod(abs(SPf),1),nfram,npat),[],2);
    Sf=zeros(nsrc,nfram);
    for p=1:npat,
        Sf(act(p,:),ind==p)=SPf(:,ind==p,p);
    end
    S(f,:,:)=reshape(Sf.',1,nfram,nsrc);
end

return;