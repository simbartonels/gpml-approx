function testDegInf()
    testdA();
    testdLLH();
end

function testdLLH()
    [x, y, ~, hyp] = initEnv();
    hyp.cov = hyp.cov(1:2);
    m = 3;
    D = size(x, 2);
    S = initSS(m, D);
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    cov_deg = {@covDegenerate, {@degSS, S}};
    optfunc = @(hypx) optimfunc(hypx, hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(hyp),[],[],[],[],[],[],@unitdisk,options);
    disp('derviative check succesfully passed');
end

function testdA()
[x, y, xs, hyp] = initEnv();
%lsf2 = 1;
%lsn2 = 0
%logell = 0
%sd = 24713; %=18071
%logell = logell(1);
m = 3;
D = size(x, 2);
S = initSS(m, D);
covhyp = hyp.cov; %[logell, lsf2];
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
optfunc = @(a) computeAdA(a, S, covhyp, x, exp(2*hyp.lik));
%fplot(optfunc, [0, 2]);
%derivative check
fmincon(optfunc,...
           covhyp,[],[],[],[],[],[],@unitdisk,options);
disp('derviative check succesfully passed');
end

function [logdetA, dlogdetA] = computeAdA(a, S, covhyp, x, sn2)
covhyp = a;
Phi = degSS(S, covhyp, x);
weight_prior = degSS(S, covhyp);
SigmaInv = diag(1./weight_prior);
A = (Phi*Phi')+sn2*SigmaInv;                      % evaluate covariance matrix
L = chol(A);
logdetA = 2*sum(log(diag(L)));
if nargout > 1
    dlogdetA = covhyp;
    for k = 1:numel(covhyp)
        dSigma = degSS(S, covhyp, [], k);
        dPhi = degSS(S, covhyp, x, k);
        dA = dPhi * Phi' + Phi * dPhi' - sn2 * SigmaInv * diag(dSigma) * SigmaInv; 
        %dlogdetA(k) = ...
        %TODO: replace with above
        dlogdetA(k) = trace(solve_chol(L, dA)); % * detA / detA
    end
end
end