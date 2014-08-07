n = 5;
m = 4;
D = 3;
x = randn(n, D);
y = randn(n, 1);
xs = randn(2, D);
hyp.lik = 0;
hyp.cov = [zeros(D, 1); 0];
varargout = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y, xs)

smhyp.lik = hyp.lik;
smhyp.v = x;
smhyp.logsigma = zeros(size(smhyp.v, 1), D);
smhyp.cov = hyp.cov;
concreteCov = {@covSM, smhyp.v, smhyp.logsigma};
%should deal the same output as the full GP
varargout = gp(smhyp, @infSM, [], concreteCov, @likGauss, x, y, xs)