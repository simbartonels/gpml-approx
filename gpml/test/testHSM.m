D = 2;
M = 32;
n = 2*M.^D;
z = 5;
%a = randn(D, 1);
%b = a + randn(D, 1).^2;
b = ones(1, D);
a = -b;

x = rand(n, D) / 2;
y = randn(n, 1);
xs = rand(z, D) / 2;
hyp.lik = 0;
hyp1.lik = hyp.lik;
hyp2.lik = hyp.lik;
logls = log(0.1);
logsf2 = 0;
hyp.cov = [logls; logsf2];
hyp1.cov = [];

e1 = zeros(M, n);
e2 = zeros(M, size(xs, 1), n);

mList = [M/2, M];

for m=M/4:M/4:M
    j = 1:m;
    j = j';

    %sqrtlambda = pi*j/sqrt(sum((b-a)).^2);
    S3 = @(r) exp(logsf2)*sqrt(2*pi*(exp(logls).^D))*exp(-2*(exp(logls)*(pi*r).^2));
    %S4(r)=S3(r/(2pi))
    S4 = @(r) exp(logsf2)*sqrt(2*pi*(exp(logls).^D))*exp(-(exp(logls)*r.^2)/2);
    %see Rasmussen p.154 above (7.11)
    S5 = @(r) exp(logsf2)*sqrt(2*pi)*(exp(logls).^D)*exp(-(exp(logls)*r).^2/2);

    %S2 = @(r) sqrt(2*pi) * exp(-exp(logls)*r.^2);
    %S2 = @(r) sqrt(2*pi*exp(logls))*exp(-2*exp(logls)*(pi*r).^2);
    
    %TODO: Unfortunately the kernel is NOT a product kernel.
    %=> code below works only for D=1
    if D==1
        cov = cell(D, 1);
        for d = 1:D
            sqrtlambda = pi*j/(b(d)-a(d));
            s = S5(sqrtlambda);
            cov(d) = {{@covHSMnaive, s, a(1, d), b(1, d), d}};
        end
        varargout1 = gp(hyp1, @infExact, [], {@covProd, cov}, @likGauss, x, y, xs)
    end
    %TODO: this is wrong. j needs to be part of the sum!
    %sqrtlambda = pi*j/sqrt(sum((b-a)).^2);
    %create index matrix
    J = zeros(D, m.^D);
    for d = 1:(D-1)
        J(d, :) = repmat(reshape(repmat((1:m)', 1, m.^(D-d))', [m.^(D-d+1), 1]), m.^(d-1), 1);
    end
    J(D, :) = repmat((1:m)', m.^(D-1), 1);
    s = zeros(m.^D, 1);
    for k = 1:m.^D
        sqrtlambda = pi*sqrt(sum((J(:, k)'./(b-a)).^2));
        s(k) = S5(sqrtlambda);
    end
    hyp2.cov = [m];
    hyp2.weight_prior = s;
    cov2 = {@covDegenerate, {@degHSM, a, b}};
    [mF s2F] = gp(hyp2, @infExactDegKernel, [], cov2, @likGauss, x, y, xs)
    %varargout1 and varargout2 should be the same
    
    %the following should deal 0
    %covHSMnaive(smhyp.cov, b, 'diag')
    %covHSMnaive(smhyp.cov, a, 'diag')
    %covHSMnaive(smhyp.cov, b, a)
    
    %e1(m, :) = (covSEiso(hyp.cov, x, 'diag') - covHSMnaive(s, a, b, hyp1.cov, x, 'diag'))';
    %e2(m, :, :) = (covSEiso(hyp.cov, x, xs) - covHSMnaive(s, a, b, hyp1.cov, x, xs))';
end

[mf s2F] = gp(hyp, @infExact, [], @covSEiso, @likGauss, x, y, xs)
