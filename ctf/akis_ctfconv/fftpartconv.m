function y = fftpartconv(h, x, L)
%FFTPARTCONV Uniform partitioned convolution scheme
%   
%	h: IR
% 	x: input signal
% 	L: block size
%
% 	Written by Archontis Politis, archontis.politis@aalto.fi

lh = length(h);
lx = length(x);
ly = lh+lx-1; 	% convolution output length (irrelevant for real-time)

N = ceil(lx/L); % number of signal partitions (irrelevant for RT)
K = ceil(lh/L); % number of IR partitions

h0 = zeros(K*L,1);
h0(1:lh) = h;
H = zeros(2*L,K);
for k=1:K
    H(1:L,k) = h0((k-1)*L+(1:L));
end
H = fft(H,2*L,1);

y = zeros((N+K)*L,1);
x0 = zeros((N+K)*L,1);
x0(1:lx) = x;
X = zeros(2*L,K);
y_n_overlap = zeros(L,1);
% processing loop over signal partitions
for n=1:N+K
    x_n = zeros(2*L,1);
    x_n(1:L) = x0((n-1)*L+(1:L));
    X(:,2:K) = X(:,1:K-1);
    X(:,1) = fft(x_n,2*L);
    B = X.*H;
    y_n = ifft(sum(B,2));
    y((n-1)*L+(1:L)) = y_n(1:L) + y_n_overlap;
    y_n_overlap = y_n(L+(1:L));
end
y = y(1:ly);
