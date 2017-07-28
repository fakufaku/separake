function S=sep_binmask(X,A)

% SEP_BINMASK STFT-domain source separation via binary masking and
% projection over the source direction, given the mixing matrix for
% instantaneous mixtures or the frequency-dependent mixing matrix for
% convolutive mixtures
%
% S=sep_binmask(X,A)
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

%%% Binary masking %%%
S=zeros(nbin,nfram,nsrc);
for f=1:nbin,
    % Projection over the source directions
    Xf=reshape(X(f,:,:),nfram,nchan).';
    proj=(A(:,:,f)'./(sqrt(sum(abs(A(:,:,f)).^2)).'*ones(1,nchan)))*Xf;
    % Selection of the best source
    [val,ind]=max(abs(proj));
    % Computation of the source coefficient
    invval=(A(:,:,f)'./(sum(abs(A(:,:,f)).^2).'*ones(1,nchan)))*Xf;
    Sf=zeros(nsrc,nfram);
    Sf((0:nfram-1)*nsrc+ind)=invval((0:nfram-1)*nsrc+ind);
    S(f,:,:)=reshape(Sf.',1,nfram,nsrc);
end

return;