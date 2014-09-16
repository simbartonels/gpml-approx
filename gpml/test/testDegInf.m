function testDegInf()
testdA();
end

function testdA()
[sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv()
%lsf2 = 1;
%lsn2 = 0
%logell = 0
%sd = 24713; %=18071
rng(sd);
logell = logell(1);
m = 3;
S = initSS(m, D, exp(logell));
covhyp = [logell, lsf2];
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
optfunc = @(a) computeAdA(a, S, covhyp, x, exp(2*lsn2));
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