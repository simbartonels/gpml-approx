function testRProp()
%TESTRPROP Summary of this function goes here
%   Detailed explanation goes here
testDegHSM();
testFullGP();
%test against full GP
end

function compareResults(idx, nlZ, mF, s2F, mFT, nlZ_o, mF_o, s2F_o, mFT_o)
mF = mF(:, idx);
s2F = s2F(:, idx);
mFT = mFT(:, idx);
nlZ = nlZ(idx);
m1 = 'GPML';
m2 = 'LibGP';
checkError(mF_o, mF, m1, m2, 'mean');
checkError(s2F_o, s2F, m1, m2, 'variance');
checkError(nlZ_o, nlZ, m1, m2, 'nlZ');
checkError(mFT_o, mFT, m1, m2, 'train mean');
end

function idx = findHyperParametersUsed(nlZ_o, nlZ)
idx = 0;
temp = nlZ_o;
for i=1:size(nlZ, 1)
    if(temp > nlZ(i))
        idx = i;
        temp = nlZ(i);
    end
end
end

function testFullGP()
iters = 1;
[trainX, trainY, testX, hyp] = initEnv();
[~, theta_over_time, mF, s2F, nlZ, mFT] = rpropmex(0, iters, trainX, trainY, testX, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp));

nlZ_o = gp(hyp, @infExact, [], {@covSEard}, @likGauss, trainX, trainY);
idx = findHyperParametersUsed(nlZ_o, nlZ);
hyp = rewrap(hyp, theta_over_time(:, idx));
[mF_o, s2F_o, ~, ~, nlZ_o] = gp(hyp, @infExact, [], {@covSEard}, @likGauss, trainX, trainY, testX);
mFT_o = gp(hyp, @infExact, [], {@covSEard}, @likGauss, trainX, trainY, trainX);

compareResults(idx, nlZ, mF, s2F, mFT, nlZ_o, mF_o, s2F_o, mFT_o);
end

function testDegHSM()
%test against DegHSM2
iters = 2;
M = 4;
m = 2;
[trainX, trainY, testX, hyp] = initEnv();
D = size(trainX, 2);
if D ~= 2, error('This test needs D = 2.'); end
[~, theta_over_time, mF, s2F, nlZ, mFT] = rpropmex(0, iters, trainX, trainY, testX, 'Solin', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), M, 'Solin');

l = 4*2.2 *ones([1, D])/3
[ J, lambda ] = initHSM( m, D, l );

nlZ_o = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, m, l, J, lambda}}, @likGauss, trainX, trainY);
idx = findHyperParametersUsed(nlZ_o, nlZ);
hyp = rewrap(hyp, theta_over_time(:, idx));
[mF_o, s2F_o, ~, ~, nlZ_o, ~] = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, m, l, J, lambda}}, @likGauss, trainX, trainY, testX);
mFT_o = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, m, l, J, lambda}}, @likGauss, trainX, trainY, trainX);

%hyp.cov(1:D) = hyp.cov(1:D)/2;
compareResults(idx, nlZ, mF, s2F, mFT, nlZ_o, mF_o, s2F_o, mFT_o);
end