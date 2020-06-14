function h = sid_stft2(x, y, winsize, hopsize, filtersize)
%SID_STFT Summary of this function goes here
%   Detailed explanation goes here

lx = length(x);
ly = size(y,1);
nCH = size(y,2);

fftsize = winsize;
overlap = winsize-hopsize;
win = rectwin(winsize);

% zero pad beginning and end
x = [zeros(hopsize,1);      x; zeros(winsize,1)];
y = [zeros(hopsize,nCH);    y; zeros(winsize,nCH)];
% stft
nBins = fftsize/2+1;
X = spectrogram(x, win, overlap, fftsize);
h = zeros(filtersize,nCH);
for nch=1:nCH
    Y = spectrogram(y(:,nch), win, overlap, fftsize);
    nFramesX = size(X,2);
    nFramesY = size(Y,2);
    if nFramesX>nFramesY
        X = X(:,1:nFramesY);
    elseif nFramesX<nFramesY
        X = [X zeros(nBins,nFramesY-nFramesX)];
    end
    
    H = zeros(nBins,1);
    for nb=1:nBins
        y_nb = Y(nb,:).';
        x_nb = X(nb,:).';
        
        H(nb) = (1/(x_nb'*x_nb))*x_nb'*y_nb;
    end
    H = [H; conj(H(end-1:-1:2))];
    h0 = ifft(H);
    h(:,nch) = h0(1:filtersize);
end
