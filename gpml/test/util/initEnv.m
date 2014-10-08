function [x, y, xs, hyp] = initEnv(sd)
%INITENV Initializes often used variables.
% sd - seed for the random number generator (optional)
if nargin == 0
    sd = floor(rand(1) * 32000)
else
    disp('Using provided seed');
end
rng(sd);
n = 17;
D = 2;
x = rand(n, D);
y = randn(n, 1);
xs = rand(2, D);
logell = randn(D, 1);
lsf2 = randn(1);
lsn2 = randn(1);
hyp.lik = lsn2;
hyp.cov = [logell; lsf2];
end