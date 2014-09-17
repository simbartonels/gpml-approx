function [sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv()
%INITENV Initializes often used variables.
sd = floor(rand(1) * 32000)
n = 13;
D = 3;
x = rand(n, D);
y = randn(n, 1);
xs = rand(2, D);
logell = randn(D, 1);
lsf2 = log(randn(1)^2);
lsn2 = log(rand(1)^2);
end