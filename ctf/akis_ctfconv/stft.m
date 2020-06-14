function spectrum = stft(insig, winsize, fftsize, hopsize, winvec)
%STFT Forward short-time Fourier-transform
%   
%	insig:      signal vector
% 	winsize:    window size in samples
% 	fftsize:    size of fft (default 2*winsize)
%   hopsize:    hop size in samples (default winsize/2)
%   winvec:     custom window (default Hanning)
%
% 	Written by Archontis Politis, archontis.politis@aalto.fi

lSig = size(insig,1);
nCHin = size(insig,2);
x = 0:(winsize-1);


% defaults
if nargin<5 || isempty(winvec)
if nargin<4 || isempty(hopsize)
    hopsize = winsize/2;
    winvec = sin(x.*(pi/winsize))'.^2; % Hamming window
elseif hopsize>winsize
    warning('Hopsize longer than window size, set to window size')
    hopsize = winsize;
    winvec = ones(winsize,1);
elseif hopsize==winsize
    winvec = ones(winsize,1);
elseif hopsize<winsize
    winvec = sin(x.*(pi/winsize))'.^2; % Hamming window
end

% time-frequency processing
if nargin<3 || isempty(fftsize)
    fftsize = 2*winsize;
end
nBins = fftsize/2 + 1;
nWindows = ceil(lSig/(2*hopsize));
nFrames = 2*nWindows+1;

% zero pad the signal's start and end for STFT
insig_pad = [zeros(winsize-hopsize, nCHin); insig; zeros(nFrames*hopsize-lSig, nCHin)];

spectrum = zeros(nBins, nFrames, nCHin);

% processing loop
idx = 1;
nf = 1;

while nf <= nFrames
    % Window input and transform to frequency domain
    insig_win = winvec*ones(1,nCHin) .* insig_pad(idx+(0:winsize-1),:);
    inspec = fft(insig_win, fftsize);
    inspec = inspec(1:nBins,:); % keep up to nyquist
    spectrum(:,nf,:) = inspec;
    
    % advance sample pointer
    idx = idx + hopsize;
    nf = nf + 1;
end

end
