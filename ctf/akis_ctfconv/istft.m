function sig = istft(inspec, winsize, hopsize, lSig)
%ISTFT Inverse short-time Fourier-transform
%   
%	inspec:     signal STFT spectrum
% 	winsize:    window size in samples
%   hopsize:    hop size in samples (default winsize/2)
%
% 	Written by Archontis Politis, archontis.politis@aalto.fi

nCHin = size(inspec,3);
nFrames = size(inspec,2);
nBins = size(inspec,1);

fftsize = 2*(nBins-1);

if nargin<2 || isempty(winsize), winsize = fftsize/2; end
if nargin<3 || isempty(hopsize), hopsize = winsize/2; end
if nargin<4 || isempty(lSig), lSig = 0; end

% processing loop
idx = 1;
nf = 1;

sig = zeros(winsize-hopsize + nFrames*hopsize + fftsize-winsize, nCHin);
while nf <= nFrames
    inspec_nf = inspec(:,nf,:);
    inspec_nf = [inspec_nf; conj(inspec_nf(end-1:-1:2,:))];
    insig_nf = ifft(inspec_nf,fftsize,1);

    % overlap-add synthesis
    sig(idx+(0:fftsize-1),:) = sig(idx+(0:fftsize-1),:) + insig_nf;
    % advance sample pointer
    idx = idx + hopsize;
    nf = nf + 1;
end
sig = sig(winsize-hopsize+1:nFrames*hopsize,:);
if lSig, sig = sig(1:lSig,:); end

end
