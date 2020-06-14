fs = 24000;
tIn = 2;
tIr = 0.5;
lIn = tIn*fs;
lIr = tIr*fs;

x = randn(lIn,1);
h = zeros(lIr,1); h(1:10:end) = 1; h = h.*exp(-5*((0:lIr-1)/lIr)).';
d = conv(h,x);
snr_db = 20;
n = sqrt(mean(d.^2))*10^(-snr_db/20)*randn(lIn+lIr-1,1);
y = d+n;

filtersize = lIr/2;
winsize = 8*filtersize;
hopsize = winsize/16;
h2 = sid_stft2(x, y, winsize, hopsize, filtersize);

plot(h), hold on, plot(h2,'--r')