function convsig = ctfconv(inspec, irspec, winsize)
%CTFCONV Summary of this function goes here
%   Detailed explanation goes here

nSigFrames = size(inspec,2);
nIrFrames = size(irspec,2);
nCHir = size(irspec,3);
nCHsig = size(inspec,3);


nBins = size(inspec,1);
fftsize = 2*(nBins-1);
if nargin<3 || isempty(winsize)
    winsize = fftsize/2;
end
hopsize = winsize/2;

% processing loop
idx = 1;
nf = 1;

convsig = zeros(winsize/2 + (nSigFrames+nIrFrames)*winsize/2 + fftsize-winsize,nCHir);
S = zeros(nBins, nIrFrames);
inspec_pad = [inspec zeros(nBins,nIrFrames)];
while nf <= nSigFrames+nIrFrames
    inspec_nf = inspec_pad(:,nf);
    S(:,2:nIrFrames) = S(:,1:nIrFrames-1);
    S(:,1) = inspec_nf;
    
    convspec_nf = sum(S.*irspec,2);
    convspec_nf = [convspec_nf; conj(convspec_nf(end-1:-1:2,:))];
    convsig_nf = ifft(convspec_nf,fftsize,1);

    % overlap-add synthesis
    convsig(idx+(0:fftsize-1),:) = convsig(idx+(0:fftsize-1),:) + convsig_nf;
    % advance sample pointer
    idx = idx + hopsize;
    nf = nf + 1; 
end
convsig = convsig(winsize+1:(nSigFrames+nIrFrames)*winsize/2,:);

end
