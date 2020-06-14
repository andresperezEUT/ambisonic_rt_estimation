
L = 100;
nSig = round(9.8*L);
nIR = 2*L;
nConvSig = nSig+nIR;

sig = randn(nSig,1);
%ir  = randn(nIR, 1).*exp(-5*((0:nIR-1)/nIR)).';
ir = zeros(nIR,1); ir(1:10:end) = 1; ir = ir.*exp(-5*((0:nIR-1)/nIR)).';
convsig = fftconv(ir,sig); % plain linear convolution (through full convolution-length FFTs)

%%
winsize = 2*L;
sigspec = stft(sig, winsize); % spectrum of input
irspec  = stft(ir,  winsize); % spectrum of filter
sigconvspec = stft(convsig,  winsize); % spectrum of convolved signal
fooconvsig = istft(sigconvspec);

figure, subplot(311), plot(sig), title('signal')
subplot(312), plot(ir), title('filter impulse response')
subplot(313), plot(convsig), hold on, plot(fooconvsig,'--r'), title('convolved signal') % comparison just for STFT/iSTFT validation

%% comparison betwene partitioned convolution and CTF cirect multiplication
partconvsig = fftpartconv(ir, sig, winsize); % partitioned convolution
ctfconvsig = ctfconv(sigspec, irspec); % convolution through ctf multiplication with fftsize = 2*winsize and Hamming window
figure, subplot(311), plot(convsig), title('plain FFT convolution')
subplot(312), plot(partconvsig), title('partitioned convolution')
subplot(313), plot(ctfconvsig), title('convolution through CTF multiplication')

%% system identification (based on windowed STFT)
hopsize = winsize/2;
nBins = winsize+1;
nFrames = size(sigspec,2);
nFiltFrames = ceil(nIR/hopsize)+1;
H = zeros(nBins, nFiltFrames);
sigspec_zpad = [sigspec zeros(nBins,nFiltFrames-1)];
for nb=1:nBins
    y_nb = sigconvspec(nb,nFiltFrames-2+(1:nFrames)).';
    X_nb = zeros(nFrames,nFiltFrames);
    for nt=1:nFrames
        X_nb(nt,:) = fliplr(sigspec_zpad(nb,nt:nt+nFiltFrames-1));
    end
    H(nb,:) = pinv(X_nb)*y_nb;
end

ir_est = istft(H,winsize); 
figure, plot(ir), hold on, plot(ir_est,'--r'), title('SID with windowed STFT 50% overlap')

%% non-overlapped STFT (winsize<IR length)
winsize = L;
sigspec = stft(sig, winsize, 2*winsize, winsize); % spectrum of input
irspec  = stft(ir,  winsize, 2*winsize, winsize); % spectrum of filter
sigconvspec = stft(convsig,  winsize, 2*winsize, winsize);

fooconvsig = istft(sigconvspec, winsize, winsize);
figure, plot(convsig), hold on, plot(fooconvsig,'--r'), title('convolved signal') % comparison just for STFT/iSTFT validation

%% system identification (based on non-overlapped STFT)

nFFT = 2*winsize;
hopsize = winsize;
nBins = winsize+1;
nFrames = size(sigspec,2);
nFiltFrames = ceil(nIR/hopsize)+1;
H = zeros(nBins, nFiltFrames);
sigspec_zpad = [sigspec zeros(nBins,nFiltFrames-1)];
for nb=1:nBins
    y_nb = sigconvspec(nb,nFiltFrames-2+(1:nFrames)).';
    X_nb = zeros(nFrames,nFiltFrames);
    for nt=1:nFrames
        X_nb(nt,:) = fliplr(sigspec_zpad(nb,nt:nt+nFiltFrames-1));
    end
    H(nb,:) = pinv(X_nb)*y_nb;
end

ir_est = istft(H,winsize,winsize);
figure, plot(ir), hold on, plot(ir_est(hopsize+1:end),'--r'), title('SID with non-overlapped STFT')

%% non-overlapped STFT (winsize>=IR length)
winsize = 2*L;
sigspec = stft(sig, winsize, 2*winsize, winsize); % spectrum of input
irspec  = stft(ir,  winsize, 2*winsize, winsize); % spectrum of filter
sigconvspec = stft(convsig,  winsize, 2*winsize, winsize);

fooconvsig = istft(sigconvspec, winsize, winsize);
figure, plot(convsig), hold on, plot(fooconvsig,'--r'), title('convolved signal') % comparison just for STFT/iSTFT validation
