function [A, mask]=estmix_inst(X,nsrc)

% ESTMIX_INST Estimates the mixing matrix of an instantaneous mixture of
% sources from its two-channel STFT.
%
% [A, mask]=estmix_inst(X,nsrc)
%
% Input:
% X: nbin x nfram x 2 matrix containing STFT coefficients with nbin
% frequency bins, nfram time frames and 2 channels
% nsrc: number of sources
%
% Output:
% A: 2 x nsrc estimated mixing matrix
% mask: mask of single source and significant data points
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2008 Pau Bofill
% This software is freeware.
% If you find it useful, please cite the following reference:
% Pau Bofill, "Identifying Single Source Data for Mixing Matrix Estimation 
% in Instantaneous BSS," Technical Report UPC-DAC-RR-2008-14,
% submitted to Icann08.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors and warnings %%%
if nargin<2, error('Not enough input arguments.'); end
[nbin,nfram,nchan]=size(X);
if nbin==2*floor(nbin/2), error('The number of frequency bins must be odd.'); end
if nchan~=2, error('The number of channels must be equal to 2.'); end

%%% Default parameters %%%
% Default parameters for ssdSelect
parsS.R=0.01; 
parsS.Theta=0.05*pi/180;
% Default parameters for ssdEstimate
parsE.lambda=4;
parsE.ngp=120; 

%%% Core processing stages %%%
% Selects single source and significant data
X=reshape(X,[nbin*nfram 2]);
mask=ssdSelect(X,nbin,parsS);
% Estimates the mixing matrix
A=ssdEstimate(X(mask,:),nsrc,parsE).';

mask = reshape(mask,nbin,nfram);

return;



function mask=ssdSelect(mix,nbin,pars)

% SSDSELECT Selects time-frequency points containing a single source.
%
% mask=ssdSelect(mix,nbin,pars)
%
% Inputs:
% mix: (nbin*nfram) x 2 matrix containing the mixture STFT coefficients.
% Each set of nbin consecutive rows is the spectrum of a time frame.
% nbin: number of frequency bins of the STFT
% pars: selection parameters
%   pars.R, selects datapoints with length above fraction pars.R of 
%      the longest one.
%   pars.Theta, selects datapoints whose angular difference with the
%       next frame is less than pars.Theta.
%
% Output:
% mask: a (nbin*nfram) x 1 bit array mask indicating which data points have
% been selected


lgt=length(mix);
mix_abs=abs(mix);
mix_angles=ssdGangles(mix_abs); % radians
mix_length=sqrt(mix_abs(:,1).^2+mix_abs(:,2).^2);    

mask=ones(lgt,1);
% Select consecutive frames with angular difference less than pars.Theta
mask=mask & [(abs(mix_angles(1:lgt-nbin)-mix_angles(nbin+1:lgt))< pars.Theta); zeros(nbin,1)];
% Select datapoints with length above fraction pars.R of the longest one
maxlength=max(mix_length);
mask=mask & (mix_length > pars.R*maxlength);
mask=mask & 1; % conversion to bitarray

return;



function [A,Fields,pangles]=ssdEstimate(mix,nsrc,pars)

% SSDESTIMATE Estimates the mixing matrix over the selected time-frequency
% points.
%
% [A,Fields,pangles]=ssdEstimate(mix,nsrc,pars)
%
% Inputs:
% mix: ndata x 2 matrix containing the mixture STFT coefficients in
% selected time-frequency points
% nsrc: number of sources
% pars: selection parameters
%   pars.ngp, number of grid points for the histogram and potential 
%      function. A reasonable setting is pars.ngp=120; 
%   pars.lambda, sharpness of the kernel function: 1 very smooth, 10 quite
%      sharp. A reasonable setting is pars.lambda = 4.
%
% Output:
% A: 2 x nsrc estimated mixing matrix
% Fields: the smoothed potential field over pars.ngp grid points
% pangles: the angular grid (in radians)


mix_abs=abs(mix);
mix_angles=ssdGangles(mix_abs); % radians
mix_length=sqrt(mix_abs(:,1).^2+mix_abs(:,2).^2);    
mx=max(mix_angles);
mn=min(mix_angles);
space=(mx-mn)/(pars.ngp-2);  % ngp grid points (to be created with histc), 
                             % ngp+1 edges
pangles=(mn-space:space:mx+space)';

Whist=[];
[Nhist,Binhist]=histc(mix_angles,pangles);
if sum(Binhist==0)>0, warning('Data out of range'); end
for binnum=1:pars.ngp+1,
    Whist(binnum,1)=sum((Binhist==binnum).*mix_length);
end
Whist(pars.ngp)=Whist(pars.ngp)+Whist(pars.ngp+1); % Last bin accounts for 
                    % data matching the upper edge (help histc).
Whist(pars.ngp+1)=[];
Nhist(pars.ngp)=Nhist(pars.ngp)+Nhist(pars.ngp+1); 
Nhist(pars.ngp+1)=[];

ind=1:pars.ngp;
across=abs(repmat(ind,pars.ngp,1)-repmat(ind',1,pars.ngp));
across=space*across;

shijs=kernel(pars.lambda,mx-mn,across); 
shijs=shijs.*repmat(Whist,1,length(ind));
Fields=sum(shijs); % total field to each grid point Pj
maxpos=cgmaxs(Fields',nsrc);  % Takes the nsrc largest peaks

directions=pangles(maxpos);
unitlength=ones(size(directions));
A=[];
[A(:,2),A(:,1)]=pol2cart(directions,unitlength);
% For drawing purposes:
pangles=pangles+space/2; pangles(pars.ngp+1)=[];

return;



function field=kernel(lambda,angrange,theta)

% KERNEL Cosine kernel function, as described in the paper.
%
% field=kernel(lambda,angrange,theta)


theta=lambda*theta;
theta=abs(theta);
range=theta<angrange/2;

field(~range)=0;
field(range)=1/2+1/2*cos(pi*2*theta(range)/angrange);
field=reshape(field,size(theta));

return;



function pks=cgmaxs(vec,nsrc)

% CGMAXS Finds the largest local maxima of a column vector
%
% pks=cgmaxs(vec,nsrc)
%
% Inputs:
% vec: np x 1 vector
% nsrc: number of local maxima to find
%
% Output:
% pks: np x 1 vector with ones located at largest peaks. Plateaux and
% extrema are ignored.


np=size(vec,1);
% get slope sign differences
vec=[vec(np-1); vec(np); vec; vec(1); vec(2)];
dvec=sign(diff(vec));
dvec=[dvec(1); dvec];
ddvec=[diff(dvec); 0];
% single point peaks
epks=(ddvec==-2);
val=sort(vec(epks));
if length(val)<nsrc, warning('Found less sources than specified.'); nsrc=length(val); end
aux=(vec>=val(length(val)-nsrc+1));
pks=epks & aux;
at=[1; 2; np+3; np+4];
pks(at)=[];

return;



function angls=ssdGangles(matr)

% SSDGANGLES Transforms the rows of a matrix from cartesian coordinates to
% angles.
%
% angls=ssdGangles(matr)
%
% Inputs:
% matr: 2 x ncols matrix
%
% Output:
% angls: 1 x ncols vector of angles


warning off
angls=atan(matr(:,1)./matr(:,2)); % atan(left./right)
warning on
negative=(matr(:,2)<0);
angls(negative)=pi+angls(negative);

return;