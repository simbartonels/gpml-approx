%Comparison of random fourier features.
D = 1;
m = 500;
n = 1; %m*D+1;
z = 1;

ls = 1; %exp(randn(1));
noise = 1; %exp(randn(1));
sf2 = 1; %exp(randn(1));

x = randn(n, D) / 2;
y = randn(size(x,1), 1);
xs = randn(z, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

K = covSEiso(hyp.cov, x, xs)

%????? (nyström method vs random fourier features: ... and fast food)
R = randn([D, m]) / ls;
W = x * R;
phi = [cos(W) sin(W)]' / sqrt(m);
Wz = xs * R;
phiz = [cos(Wz) sin(Wz)]' / sqrt(m);
G = phi' * phiz
%1
sum(sum(abs((K - G).^2))/(n*z))

% phi = [cos(W) sin(W)]';
% phiz = [cos(Wz) sin(Wz)]';
% G = phi' * phiz / m;
% %1
% sum(sum(abs((K - G).^2))/(n*z))

phi = (exp(1i*W')/sqrt(m));
phiz = (exp(-1i*Wz')/sqrt(m));
%this does not work because phi' is the complex conjugate and not just
%transpose
G = phi' * phiz;
%2
sum(sum(abs((K - G).^2))/(n*z))

%fast food
phiz = (exp(1i*W')/sqrt(m));
G = phi' * phiz;
%3
sum(sum(abs((K - G).^2))/(n*z))

%random features for large scale kernel machines
b = 2*pi*rand([m, 1]);
phi = sqrt(2)*cos(W' + repmat(b, [1, n]));
phiz = sqrt(2)*cos(Wz' + repmat(b, [1, n]));
G = phi' * phiz / m;
%4
sum(sum(abs((K - G).^2))/(n*z))

%uniform approximation of functions with random bases
b = 2*pi*rand([m, 1])-pi;
phi = cos(W' + repmat(b, [1, n])) / sqrt(m);
phiz = cos(Wz' + repmat(b, [1, n])) / sqrt(m);
G = phi' * phiz;
%5
sum(sum(abs((K - G).^2))/(n*z))

%this is from the source code: see rp_projections.m
R = sqrt(2)*R';
phi = exp(1i*R*x)/sqrt(m);
phiz = exp(1i*R*xs)/sqrt(m);
G = phi' * phiz;
%6
sum(sum(abs((K - G).^2))/(n*z))
