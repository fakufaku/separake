function I=src_image(S,A)

% SRC_IMAGE STFT-domain projection of sources into the mixture domain,
% given the mixing matrix for instantaneous mixtures or the
% frequency-dependent mixing matrix for convolutive mixtures
%
% I=src_image(S,A)
%
% Inputs:
% S: nbin x nfram x nsrc matrix containing the source STFT coefficients
% A: either a nchan x nsrc mixing matrix (for instantaneous mixtures) or a
% nchan x nsrc x nbin frequency-dependent mixing matrix (for convolutive
% mixtures)
%
% Output:
% I: nbin x nfram x nsrc x nchan matrix containing the STFT coefficients of
% the spatial source images
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2008 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors and warnings %%%
if nargin<2, error('Not enough input arguments.'); end
[nbin,nfram,nsrc]=size(S);
if nbin==2*floor(nbin/2), error('The number of frequency bins must be odd.'); end
[nchan,nsrc2,nbin2]=size(A);
if nsrc2~=nsrc, error('The number of sources is different within the source matrix and the mixing matrix.'); end
if nbin2==1,
    A=repmat(A,[1 1 nbin]);
elseif nbin2~=nbin,
    error('The number of frequency bins is different within the sources and the frequency-dependent mixing matrix.');
end

%%% Filtering %%%
I=zeros(nbin,nfram,nsrc,nchan);
for j=1:nsrc,
    for f=1:nbin,
        Ijf=A(:,j,f)*S(f,:,j);
        I(f,:,j,:)=reshape(Ijf.',1,nfram,1,nchan);
    end
end

return;